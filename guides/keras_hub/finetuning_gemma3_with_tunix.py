"""
Title: Fine-tuning a KerasHub Gemma3 checkpoint with Tunix
Author: [Minho Ryu](https://github.com/bzantium)
Date created: 2026/11/09
Last modified: 2026/11/09
Description: Export Gemma3 270M from KerasHub and verify a Tunix training roundtrip.
Accelerator: None
"""

"""
## Introduction

This guide starts with the pretrained **Gemma3 270M IT** model in KerasHub,
exports its weights to Hugging Face safetensors, and loads them into Tunix.
We compare logits and parameter counts, run two supervised fine-tuning steps,
and verify that saving and restoring the trained parameters preserves output.

KerasHub supplies the checkpoint and export API. Tunix runs its own Flax NNX
Gemma3 implementation and `PeftTrainer`; the Keras model itself is not passed
to the trainer. This workflow uses the existing APIs in both libraries.

The example uses one CPU device and float32. It needs several GB of RAM for
model copies, gradients, and checkpoint validation. GPU/TPU execution and
importing the trained weights back into KerasHub are outside this guide.
"""


"""
## Setup

Use a fresh Python 3.12 environment. These optional dependency pins reproduce
the CPU environment used for this guide. The KerasHub source revision pins the
exporter implementation used for verification.
"""

"""shell
pip install "keras-hub @ git+https://github.com/keras-team/keras-hub.git@7ff25e361a42e923def75351a022778d0480e6e6"
pip install "keras==3.15.1" "google-tunix==0.1.7" "jax==0.11.0" "flax==0.12.8" "safetensors==0.7.0" "datasets==4.4.1" "huggingface-hub==0.36.2" "gcsfs==2025.10.0"
"""


"""
Accept the Gemma model terms on
[Hugging Face](https://huggingface.co/google/gemma-3-270m-it) and authenticate
with an account that can download the weights (`hf auth login` in a terminal).
In a notebook, use `huggingface_hub.notebook_login()` before running the model
cells. Keep tokens out of source code and saved outputs.

Set the backend before importing Keras. Restart the runtime if Keras or JAX
was already imported with different settings.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

import gc
import importlib.metadata
import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import keras
import keras_hub
import numpy as np
import optax
from flax import nnx
from orbax import checkpoint as ocp
from tunix.models.gemma3 import model as gemma3
from tunix.models.gemma3 import params_safetensors
from tunix.sft import peft_trainer
from tunix.sft import utils

"""
## Prepare model inputs and measure loss

Tunix takes token IDs, positions, a causal attention mask, and a separate
loss mask. Positions and attention masks come from the input tokens; the loss
mask selects only the completion. A batch with no next-token targets is an
input error, so the helper rejects it before evaluating the model.
"""


def model_inputs(batch):
    mask = batch.input_tokens != 0
    return {
        "input_tokens": batch.input_tokens,
        "input_mask": batch.input_mask,
        "positions": utils.build_positions_from_mask(mask),
        "attention_mask": utils.make_causal_attn_mask(mask),
    }


def logits(model, batch):
    inputs = model_inputs(batch)
    values, _ = model(
        inputs["input_tokens"],
        inputs["positions"],
        None,
        inputs["attention_mask"],
    )
    return values


def loss(model, batch):
    weights = batch.input_mask[:, 1:]
    num_targets = int(jnp.sum(weights))
    if num_targets == 0:
        raise ValueError("The batch must contain at least one target token.")
    values = logits(model, batch)[:, :-1].astype(jnp.float32)
    targets = batch.input_tokens[:, 1:]
    losses = optax.softmax_cross_entropy_with_integer_labels(values, targets)
    return float(jnp.sum(losses * weights) / num_targets)


"""
## Load and export the pretrained checkpoint

The only training example is the instruction **"Say hello."** with completion
**"Hello!"**. Prompt tokens are visible to attention but excluded from loss;
completion and turn-ending tokens contribute to the objective.

Loss will be measured on this same batch before and after training. Its
reduction checks that gradients and parameter updates work; it does not measure
held-out performance or establish a fine-tuning recipe.
"""

preset = "hf://google/gemma-3-270m-it"
output_dir = Path(tempfile.mkdtemp(prefix="gemma3-tunix-"))
steps = 2
learning_rate = 1e-5
logit_atol = 1e-3

keras.utils.set_random_seed(42)
output_dir.mkdir(parents=True, exist_ok=True)
export_dir = output_dir / "export"
keras_model = keras_hub.models.Gemma3CausalLM.from_preset(preset, dtype="float32")
backbone = keras_model.backbone
keras_model.export_to_transformers(str(export_dir))

# Prompt tokens are inputs, while only completion tokens contribute
# to the supervised loss. This is a plumbing demo, not a benchmark.
tokenizer = keras_model.preprocessor.tokenizer
prompt = tokenizer(
    "<start_of_turn>user\nSay hello.<end_of_turn>\n<start_of_turn>model\n"
)
completion = tokenizer("Hello!<end_of_turn>")
prompt = np.asarray(prompt, dtype="int32").reshape(-1)
completion = np.asarray(completion, dtype="int32").reshape(-1)
tokens = np.concatenate(([tokenizer.start_token_id], prompt, completion))[None].astype(
    "int32"
)
mask = np.zeros_like(tokens, dtype="bool")
mask[:, 1 + len(prompt) :] = True
batch = peft_trainer.TrainingInput(input_tokens=tokens, input_mask=mask)
keras_logits = np.asarray(
    keras.ops.convert_to_numpy(
        keras_model(
            {
                "token_ids": tokens,
                "padding_mask": np.ones_like(tokens),
            }
        )
    )
)

config = gemma3.ModelConfig(
    num_layers=backbone.num_layers,
    num_embed=backbone.vocabulary_size,
    embed_dim=backbone.hidden_dim,
    hidden_dim=backbone.intermediate_dim,
    num_heads=backbone.num_query_heads,
    head_dim=backbone.head_dim,
    num_kv_heads=backbone.num_key_value_heads,
    sliding_window_size=backbone.sliding_window_size,
    local_base_frequency=10_000,
    global_base_frequency=1_000_000,
    param_dtype=jnp.float32,
)
keras_parameter_count = backbone.count_params()
del keras_model, backbone
gc.collect()
mesh = jax.make_mesh(
    (1, 1),
    ("fsdp", "tp"),
    axis_types=(jax.sharding.AxisType.Auto,) * 2,
)


"""
## Load the export in Tunix and compare logits

The Tunix configuration mirrors the KerasHub backbone dimensions. Both
implementations receive identical token IDs and unpadded input. The one-device
mesh retains Tunix's named sharding axes without requiring a distributed setup.

We require equal parameter counts, logits within `atol=1e-3, rtol=1e-4`, and
identical argmax tokens at every input position. The measured maximum/mean
absolute errors on this CPU run were `3.28e-4` / `4.96e-5`.

The absolute tolerance allows float32 arithmetic differences between the model
implementations. As another reference, a Transformers float32 eager-attention
model loaded from the same export differed by at most `8.24e-4` from KerasHub
and `6.52e-4` from Tunix, with matching argmax tokens. These measurements cover
this input and environment; the assertions below still run on every execution.
"""

with jax.set_mesh(mesh):
    tunix_model = params_safetensors.create_model_from_safe_tensors(
        str(export_dir), config, mesh, dtype=jnp.float32
    )
    before = np.asarray(logits(tunix_model, batch))
    difference = np.abs(keras_logits - before)
    print(
        f"Logits: max abs diff={difference.max():.6g}, "
        f"mean abs diff={difference.mean():.6g}",
        flush=True,
    )
    np.testing.assert_allclose(keras_logits, before, atol=logit_atol, rtol=1e-4)
    argmax_agreement = float(np.mean(keras_logits.argmax(-1) == before.argmax(-1)))
    assert argmax_agreement == 1.0
    state = nnx.state(tunix_model, nnx.Param)
    parameter_count = sum(x.size for x in jax.tree.leaves(state))
    assert parameter_count == keras_parameter_count


"""
## Run two supervised fine-tuning steps

`PeftTrainer` also supports full fine-tuning: this example passes the complete
model with an SGD optimizer. We use a conservative `1e-5` learning rate.
Besides checking finite, decreasing same-batch loss, we explicitly compare
parameters to ensure that training changed them.

Discard the KerasHub model and comparison snapshots once they are no longer
needed, to leave memory for training and checkpoint restoration.
"""

with jax.set_mesh(mesh):
    original_parameters = [np.array(x) for x in jax.tree.leaves(state)]
    initial_loss = loss(tunix_model, batch)
    trainer = peft_trainer.PeftTrainer(
        model=tunix_model,
        optimizer=optax.sgd(learning_rate=learning_rate),
        training_config=peft_trainer.TrainingConfig(
            eval_every_n_steps=steps + 1,
            max_steps=steps,
        ),
    ).with_gen_model_input_fn(model_inputs)
    trainer.train(train_ds=[batch] * steps)
    final_loss = loss(tunix_model, batch)
    after = np.asarray(logits(tunix_model, batch))
    assert np.isfinite(initial_loss) and np.isfinite(final_loss)
    assert not np.array_equal(before, after), "Training did not change logits."
    assert final_loss < initial_loss, (initial_loss, final_loss)
    trained_state = nnx.state(tunix_model, nnx.Param)
    parameter_change = max(
        float(np.max(np.abs(old - np.asarray(new))))
        for old, new in zip(original_parameters, jax.tree.leaves(trained_state))
    )
    assert parameter_change > 0, "Training did not update parameters."
    del original_parameters, state, trainer
    gc.collect()


"""
## Save and restore the trained parameters

The original safetensors export describes the base model. Store the trained
NNX parameter state separately with Orbax, construct another Tunix model from
the original export, and restore the trained state into it. Exact logit equality
checks that the trained state survived the roundtrip. This does not convert
the trained parameters back into KerasHub format.
"""

with jax.set_mesh(mesh):
    # Save the trained NNX parameters separately from the original export.
    checkpoint_path = str((output_dir / "trained_parameters").resolve())
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(checkpoint_path, trained_state, force=True)
        checkpointer.wait_until_finished()
        reloaded = params_safetensors.create_model_from_safe_tensors(
            str(export_dir), config, mesh, dtype=jnp.float32
        )
        restored = checkpointer.restore(
            checkpoint_path, target=nnx.state(reloaded, nnx.Param)
        )
        nnx.update(reloaded, restored)
        restored_logits = np.asarray(logits(reloaded, batch))
    np.testing.assert_array_equal(after, restored_logits)


"""
## Record the results

The CPU run matched **268,098,176** parameters. With this single short training
example, loss fell from `4.57253` to `0.01496` after two steps. A low loss on
that same example is expected from memorization and is not a quality metric.

The JSON report includes actual numerical errors, optimizer settings, package
versions, and the device. Both the original export and trained parameters stay
in the temporary output directory for inspection.
"""

results = {
    "preset": preset,
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("keras", "google-tunix", "jax", "flax", "safetensors")
    },
    "devices": [str(device) for device in jax.devices()],
    "parameters": parameter_count,
    "max_absolute_logit_error": float(difference.max()),
    "mean_absolute_logit_error": float(difference.mean()),
    "logit_atol": logit_atol,
    "logit_rtol": 1e-4,
    "argmax_agreement": argmax_agreement,
    "training_steps": steps,
    "learning_rate": learning_rate,
    "initial_loss": initial_loss,
    "final_loss": final_loss,
    "max_logit_change_after_training": float(np.max(np.abs(after - before))),
    "max_parameter_change": parameter_change,
    "trained_checkpoint_logits_match": True,
}
(output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))

"""
## Next steps

For a meaningful fine-tuning experiment, use multiple examples, a separate
validation split, and evaluate generated responses on unseen prompts. This
guide covers one text-only Gemma3 checkpoint, unpadded input, full fine-tuning,
and a single CPU device. Gemma4, LoRA, distributed training, and a KerasHub
import path for the trained weights require separate validation.
"""
