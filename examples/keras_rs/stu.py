"""
Title: Implementing a Sequential Transduction Unit (STU) layer for Keras-RS
Author: (Lakshmi Kala Kadali)[https://github.com/lakshmikala]
Date created: 2025/09/25
Last modified: 2025/09/25
Description: A tutorial on using and understanding a custom Keras STU Layer.
Accelerator: GPU
"""

"""
## Introduction
This tutorial will guide you through understanding and using a custom Keras layer called
the Structured Transformer Unit (STU). While traditional Transformers rely on fixed-length
sequences and often use padding, the STU is designed to handle variable-length sequences
efficiently through the use of jagged tensors.

The key benefits of this approach are:
Memory Efficiency: No need to pad short sequences with empty values.
Computational Efficiency: Operations are performed only on real data, avoiding wasted
computation on padding.

This tutorial focuses on the implementation of the core components of the STU and
how they interact to form a functional Keras layer. We'll explore how jagged tensors
are handled, how a custom attention mechanism is built, and how the entire logic is
wrapped into a reusable 'keras.layers.Layer' class.

"""

"""shell
!pip install -q keras-rs
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # `"jax"`/`"torch"`

import keras
from typing import List, Optional, Tuple
from keras import ops, layers



# Common and Masking Utilities ---

"""
The fx_unwrap_optional_tensor function is primarily designed to manage the optional
state (like the Key-Value cache) within the custom Keras layers.
"""


def fx_unwrap_optional_tensor(
    optional: Optional[keras.KerasTensor],
) -> keras.KerasTensor:
    # Helper to unwrap optional tensors, returning a zero-tensor for uninitialized cache.
    if optional is None:
        return ops.zeros((0,), dtype="float32")
    return optional


"""
This function, get_valid_attn_mask_keras, is a highly sophisticated utility designed to
generate a 3D attention mask M∈{0,1} that enforces various constraints (causality,
maximum attention length, sequence padding, and context length) during a Multi-Head Attention
(MHA) operation.
"""


def get_valid_attn_mask_keras(
    causal: bool,
    N: int,
    seq_lengths: keras.KerasTensor,
    num_targets: Optional[keras.KerasTensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> keras.KerasTensor:
    ids = ops.reshape(ops.arange(0, N, dtype="int32"), (1, N))
    max_ids = ops.reshape(seq_lengths, (-1, 1, 1))
    B = ops.shape(seq_lengths)[0]
    if contextual_seq_len > 0:
        ids = ops.maximum(ids - contextual_seq_len + 1, 0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - ops.reshape(num_targets, (-1, 1, 1))
        ids = ops.minimum(ids, max_ids)
        row_ids = ops.broadcast_to(ops.reshape(ids, (-1, N, 1)), (B, N, N))
        col_ids = ops.broadcast_to(ops.reshape(ids, (-1, 1, N)), (B, N, N))
    else:
        row_ids = ops.broadcast_to(ops.reshape(ids, (N, 1)), (N, N))
        col_ids = ops.transpose(row_ids)
        row_ids = ops.reshape(row_ids, (1, N, N))
        col_ids = ops.reshape(col_ids, (1, N, N))
        max_ids = None
    row_col_dist = row_ids - col_ids
    valid_attn_mask = ops.reshape(ops.eye(N, dtype="bool"), (1, N, N))
    if not causal:
        row_col_dist = ops.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = ops.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        valid_attn_mask = ops.logical_and(valid_attn_mask, row_col_dist <= max_attn_len)
    if contextual_seq_len > 0 and max_ids is not None:
        valid_attn_mask = ops.logical_or(
            valid_attn_mask, ops.logical_and(row_ids == 0, col_ids < max_ids)
        )
    return valid_attn_mask


"""
## Jagged Tensors
A jagged tensor is a representation of a list of sequences, where each sequence can have
a different length. It's composed of two parts:
values: A single, flattened tensor containing all the elements from all sequences.
offsets: A tensor of indices that specifies where each sequence starts in the values tensor.
To perform standard matrix operations on jagged tensors, we must first convert them into a
regular, padded dense format. The code provides utility functions for this.
"""


def keras_jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
    ##Converts a flattened jagged tensor to a padded dense tensor [B, N, D_flat].
    offsets = offsets[0] if isinstance(offsets, list) else offsets
    B = ops.shape(offsets)[0] - 1
    max_len = max_lengths[0]
    D_flat = ops.shape(values)[-1]
    if ops.shape(values)[0] == 0:
        return ops.full((B, max_len, D_flat), padding_value, dtype=values.dtype)

    def pad_one(i):
        start = offsets[i]
        end = offsets[i + 1]
        seq_len = end - start
        seq = ops.slice(values, [start, 0], [seq_len, D_flat])
        if ops.equal(seq_len, 0):
            return ops.full((max_len, D_flat), padding_value, dtype=values.dtype)
        if seq_len < max_len:
            padding_shape = ops.stack([max_len - seq_len, D_flat])
            padding = ops.full(padding_shape, padding_value, dtype=values.dtype)
            return ops.concatenate([seq, padding], axis=0)
        else:
            return seq[:max_len]

    idxs = ops.arange(B, dtype="int32")
    return ops.map(pad_one, idxs)


def keras_dense_to_jagged(
    dense: keras.KerasTensor,
    x_offsets: List[keras.KerasTensor],
) -> keras.KerasTensor:
    ##Converts a padded dense tensor [B, N, D] back into a jagged tensor [L, D].
    seq_offsets = x_offsets[0]
    N = ops.shape(dense)[1]
    D_flat = ops.shape(dense)[2]
    token_range = ops.arange(N)
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    mask = ops.expand_dims(token_range, axis=0) < ops.expand_dims(seq_lengths, axis=1)
    flattened = ops.reshape(dense, [-1, D_flat])
    flattened_mask = ops.reshape(mask, [-1])
    return flattened[flattened_mask]


def split_2D_jagged(
    max_seq_len: int,
    values: keras.KerasTensor,
    total_len_left: Optional[int] = None,
    total_len_right: Optional[int] = None,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[keras.KerasTensor] = None,
    offsets_right: Optional[keras.KerasTensor] = None,
    kernel=None,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    def keras_split_2D_jagged_jagged(max_seq_len, values, offsets_left, offsets_right):
        D_flat = ops.shape(values)[1]
        offsets = offsets_left + offsets_right
        padded_values_bnd = keras_jagged_to_padded_dense(
            values=values,
            offsets=[offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        padded_values = ops.reshape(padded_values_bnd, [-1, D_flat])
        lengths_left = offsets_left[1:] - offsets_left[:-1]
        lengths_right = offsets_right[1:] - offsets_right[:-1]
        mask = ops.reshape(ops.arange(max_seq_len, dtype="int32"), [1, -1])
        lengths_left_broadcast = ops.reshape(lengths_left, [-1, 1])
        lengths_right_combined = ops.reshape(lengths_left + lengths_right, [-1, 1])
        mask_left = mask < lengths_left_broadcast
        mask_right = ops.logical_and(
            mask >= lengths_left_broadcast, mask < lengths_right_combined
        )
        return (
            padded_values[ops.reshape(mask_left, [-1])],
            padded_values[ops.reshape(mask_right, [-1])],
        )

    L_total = ops.shape(values)[0]
    offsets_left_non_optional = offsets_left
    if offsets_left is None:
        offsets_left_non_optional = max_len_left * ops.arange(
            L_total // max_len_left + 1, dtype="int32"
        )
    offsets_right_non_optional = offsets_right
    if offsets_right is None:
        offsets_right_non_optional = max_len_right * ops.arange(
            L_total // max_len_right + 1, dtype="int32"
        )
    return keras_split_2D_jagged_jagged(
        max_seq_len=max_seq_len,
        values=values,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def concat_2D_jagged(
    max_seq_len: int,
    values_left: keras.KerasTensor,
    values_right: keras.KerasTensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[keras.KerasTensor] = None,
    offsets_right: Optional[keras.KerasTensor] = None,
    kernel=None,
) -> keras.KerasTensor:
    def keras_concat_2D_jagged_jagged(
        values_left,
        values_right,
        max_len_left,
        max_len_right,
        offsets_left,
        offsets_right,
    ):
        max_seq_len = max_len_left + max_len_right
        lengths_left = offsets_left[1:] - offsets_left[:-1]
        lengths_right = offsets_right[1:] - offsets_right[:-1]
        padded_left = keras_jagged_to_padded_dense(
            values=values_left,
            offsets=[offsets_left],
            max_lengths=[max_len_left],
            padding_value=0.0,
        )
        padded_right = keras_jagged_to_padded_dense(
            values=values_right,
            offsets=[offsets_right],
            max_lengths=[max_len_right],
            padding_value=0.0,
        )
        concatted_dense = ops.concatenate([padded_left, padded_right], axis=1)
        lengths_left_broadcast = ops.reshape(lengths_left, [-1, 1])
        lengths_right_broadcast = ops.reshape(lengths_right, [-1, 1])
        mask = ops.reshape(ops.arange(max_seq_len, dtype="int32"), [1, -1])
        mask = ops.logical_or(
            mask < lengths_left_broadcast,
            ops.logical_and(
                mask >= max_len_left, mask < max_len_left + lengths_right_broadcast
            ),
        )
        return concatted_dense[ops.reshape(mask, [-1])]

    def keras_concat_2D_jagged_resolver(
        values_left,
        values_right,
        max_len_left,
        max_len_right,
        offsets_left,
        offsets_right,
    ):
        L_total = ops.shape(values_left)[0]
        offsets_left_non_optional = offsets_left
        if offsets_left is None:
            offsets_left_non_optional = max_len_left * ops.arange(
                L_total // max_len_left + 1, dtype="int32"
            )
        offsets_right_non_optional = offsets_right
        if offsets_right is None:
            offsets_right_non_optional = max_len_right * ops.arange(
                L_total // max_len_right + 1, dtype="int32"
            )
        if max_len_left is None:
            max_len_left_final = ops.max(
                offsets_left_non_optional[1:] - offsets_left_non_optional[:-1]
            )
        else:
            max_len_left_final = max_len_left
        if max_len_right is None:
            max_len_right_final = ops.max(
                offsets_right_non_optional[1:] - offsets_right_non_optional[:-1]
            )
        else:
            max_len_right_final = max_len_right
        return keras_concat_2D_jagged_jagged(
            values_left=values_left,
            values_right=values_right,
            max_len_left=max_len_left_final,
            max_len_right=max_len_right_final,
            offsets_left=offsets_left_non_optional,
            offsets_right=offsets_right_non_optional,
        )

    return keras_concat_2D_jagged_resolver(
        values_left=values_left,
        values_right=values_right,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left,
        offsets_right=offsets_right,
    )


# --- Compute and Output Utilities ---

"""This Keras function, keras_layer_norm, implements the steps of Layer Normalization in a
functional, step-by-step manner, rather than using a built-in Keras layer.
"""


def keras_layer_norm(x, weight, bias, eps):
    # Functional Layer Norm steps
    normalized_x = ops.layer_norm(x, axis=-1, epsilon=eps)
    return normalized_x * weight + bias


"""This simple Keras function, keras_addmm, implements the fundamental operation of an
affine transformation or a fully-connected layer in neural networks.
"""


def keras_addmm(bias, input, mat2):
    return ops.add(bias, ops.matmul(input, mat2))


"""This Keras function, keras_norm_mul_dropout, implements a crucial non-linear
transformation block used in architectures like the Sequential Transduction Unit (STU).
It combines Layer Normalization, a gating mechanism using the u vector, and Dropout
"""


def keras_norm_mul_dropout(
    x,
    u,
    weight,
    bias,
    eps,
    dropout_ratio,
    training,
    silu_u=False,
    concat_ux=False,
    group_norm=False,
    num_heads=1,
    linear_dim=-1,
):
    x = ops.convert_to_tensor(x, dtype="float32")
    u = ops.convert_to_tensor(u, dtype="float32")
    if silu_u:
        u = ops.silu(u)
    if group_norm:
        raise NotImplementedError(
            "Group Norm path not suitable for simple Keras ops conversion."
        )
    else:
        y_norm = ops.layer_norm(x, axis=-1, epsilon=eps) * weight + bias
        y = u * y_norm
    if concat_ux:
        y = ops.concatenate([u, x, y], axis=1)
    y = keras.layers.Dropout(dropout_ratio)(y, training=training)
    return ops.cast(y, dtype=x.dtype)


"""The function hstu_compute_uqvk is a core preprocessing step in the Sequential Transduction
Unit (STU). It takes the layer's input, applies normalization, and performs a single large
linear projection to generate four essential feature vectors: the standard q,k,v for
attention, and a unique u vector for gated feature control.
"""


def hstu_compute_uqvk(
    x,
    norm_weight,
    norm_bias,
    norm_eps,
    num_heads,
    attn_dim,
    hidden_dim,
    uvqk_weight,
    uvqk_bias,
    kernel=None,
):
    normed_x = keras_layer_norm(x, weight=norm_weight, bias=norm_bias, eps=norm_eps)
    uvqk = keras_addmm(uvqk_bias, normed_x, uvqk_weight)
    u_size = hidden_dim * num_heads
    v_size = hidden_dim * num_heads
    q_size = attn_dim * num_heads
    k_size = attn_dim * num_heads
    start_u = 0
    start_v = u_size
    start_q = u_size + v_size
    start_k = u_size + v_size + q_size
    L_out = ops.shape(uvqk)[0]
    u = ops.slice(uvqk, start_indices=[0, start_u], shape=[L_out, u_size])
    v = ops.slice(uvqk, start_indices=[0, start_v], shape=[L_out, v_size])
    q = ops.slice(uvqk, start_indices=[0, start_q], shape=[L_out, q_size])
    k = ops.slice(uvqk, start_indices=[0, start_k], shape=[L_out, k_size])
    u = ops.silu(u)
    q = ops.reshape(q, [-1, num_heads, attn_dim])
    k = ops.reshape(k, [-1, num_heads, attn_dim])
    v = ops.reshape(v, [-1, num_heads, hidden_dim])
    return u, q, k, v


"""The hstu_compute_output function concludes the forward pass of the Sequential Transduction
Unit (STU) by combining the attention mechanism's output with the unique gate vector(u)
and applying the residual connection. This final stage transforms the processed features
back into the model's main embedding dimension.
"""


def hstu_compute_output(
    attn,
    u,
    x,
    norm_weight,
    norm_bias,
    norm_eps,
    output_weight,
    num_heads,
    linear_dim,
    dropout_ratio,
    training,
    concat_ux,
    group_norm,
    recompute_y_in_backward,
):
    y = keras_norm_mul_dropout(
        x=attn,
        u=u,
        weight=norm_weight,
        bias=norm_bias,
        eps=norm_eps,
        dropout_ratio=dropout_ratio,
        training=training,
        silu_u=False,
        concat_ux=concat_ux,
        group_norm=group_norm,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )
    output = ops.add(x, ops.matmul(y, output_weight))
    return output


# --- Attention Kernels ---

"""The keras_pad_qkv function is a crucial step in the Sequential Transduction Unit (STU),
acting as the bridge between the jagged tensor format and the standard tensor shape required
for the Multi-Head Attention (MHA) dot product. It converts the q,k,v feature vectors from
their memory-efficient, flattened format into a padded, batch-major format.
"""


def keras_pad_qkv(q, k, v, seq_offsets, N):
    L, H, D = ops.shape(q)
    V_dim = ops.shape(v)[2]
    values_q_k = ops.reshape(q, [L, H * D])
    values_v = ops.reshape(v, [L, H * V_dim])
    padded_q_k = keras_jagged_to_padded_dense(
        values=values_q_k, offsets=[seq_offsets], max_lengths=[N], padding_value=0.0
    )
    padded_v = keras_jagged_to_padded_dense(
        values=values_v, offsets=[seq_offsets], max_lengths=[N], padding_value=0.0
    )
    B = ops.shape(padded_q_k)[0]
    padded_q_k = ops.reshape(padded_q_k, [B, N, H, D])
    padded_v = ops.reshape(padded_v, [B, N, H, V_dim])
    padded_q = ops.transpose(padded_q_k, [0, 2, 1, 3])
    padded_k = ops.transpose(padded_q_k, [0, 2, 1, 3])
    padded_v = ops.transpose(padded_v, [0, 2, 1, 3])
    return padded_q, padded_k, padded_v


"""The keras_hstu_mha function is the core Multi-Head Attention (MHA) implementation within
the Sequential Transduction Unit (STU). It processes the jagged Query, Key, and Value tensors,
performs the attention mechanism, enforces masking, and returns the result back in the
efficient jagged format.
"""


def keras_hstu_mha(
    q,
    k,
    v,
    seq_offsets,
    max_seq_len,
    alpha,
    causal=True,
    dropout_pr=0.0,
    training=True,
    attn_scale=None,
    **kwargs
):
    L, H, _ = ops.shape(q)
    V_dim = ops.shape(v)[2]
    q, k, v = keras_pad_qkv(q, k, v, seq_offsets, max_seq_len)
    qk_attn = ops.einsum("bhxa,bhya->bhxy", q, k) * alpha
    if attn_scale is not None:
        if ops.ndim(attn_scale) > 0:
            attn_scale_padded = keras_jagged_to_padded_dense(
                values=ops.expand_dims(attn_scale, axis=-1),
                offsets=[seq_offsets],
                max_lengths=[max_seq_len],
                padding_value=0.0,
            )
            attn_scale_padded = ops.expand_dims(
                ops.cast(attn_scale_padded, qk_attn.dtype), axis=1
            )
        qk_attn = ops.silu(qk_attn) * attn_scale_padded
    else:
        qk_attn = ops.silu(qk_attn) / max_seq_len
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    valid_attn_mask = get_valid_attn_mask_keras(
        causal=causal, N=max_seq_len, seq_lengths=seq_lengths, **kwargs
    )
    qk_attn = qk_attn * ops.expand_dims(
        ops.cast(valid_attn_mask, qk_attn.dtype), axis=1
    )
    if dropout_pr > 0.0 and training:
        qk_attn = keras.layers.Dropout(dropout_pr)(qk_attn, training=training)
    attn_dense = ops.einsum("bhxd,bhdv->bhxv", qk_attn, v)
    flat_attn_dense = ops.reshape(
        ops.transpose(attn_dense, [0, 2, 1, 3]), [-1, max_seq_len, H * V_dim]
    )
    jagged_output = keras_dense_to_jagged(flat_attn_dense, [seq_offsets])
    L_out = ops.shape(jagged_output)[0]
    return ops.reshape(jagged_output, [L_out, H, V_dim])


"""The keras_cached_hstu_mha function is the specialized attention mechanism used for
efficient incremental inference in the Structured Transformer Unit (STU).
It avoids re-computing attention scores for the entire sequence by only calculating
the interaction between the newly generated token's query and the full Key/Value cache.
"""


def keras_cached_hstu_mha(
    max_seq_len,
    alpha,
    delta_q,
    k,
    v,
    seq_offsets,
    num_targets=None,
    max_attn_len=0,
    contextual_seq_len=0,
    enable_tma=False,
):
    L_delta, H, D = ops.shape(delta_q)
    B = ops.shape(seq_offsets)[0] - 1
    DeltaSize = L_delta // B
    V_dim = ops.shape(v)[2]
    delta_q = ops.transpose(
        ops.reshape(delta_q, (B, DeltaSize, H, D)), perm=[0, 2, 1, 3]
    )
    N_full = max_seq_len
    k_full = ops.transpose(ops.reshape(k, (B, N_full, H, D)), [0, 2, 1, 3])
    v_full = ops.transpose(ops.reshape(v, (B, N_full, H, V_dim)), [0, 2, 1, 3])
    qk_attn = ops.einsum("bhxa,bhya->bhxy", delta_q, k_full) * alpha
    qk_attn = ops.silu(qk_attn) / max_seq_len
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    full_valid_attn_mask = get_valid_attn_mask_keras(
        causal=True,
        N=max_seq_len,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    valid_attn_mask_sliced = full_valid_attn_mask[:, -DeltaSize:, :]
    qk_attn = qk_attn * ops.expand_dims(
        ops.cast(valid_attn_mask_sliced, qk_attn.dtype), axis=1
    )
    attn_output = ops.einsum("bhxd,bhdv->bhxv", qk_attn, v_full)
    attn_output = ops.transpose(attn_output, perm=[0, 2, 1, 3])
    return ops.reshape(attn_output, (-1, H, V_dim))


##wrapper for delta_hstu_mha to call keras_cached_hstu_mha
def delta_hstu_mha(
    max_seq_len,
    alpha,
    delta_q,
    k,
    v,
    seq_offsets,
    num_targets=None,
    max_attn_len=0,
    contextual_seq_len=0,
    kernel=None,
    enable_tma=False,
):
    L_delta, H, D = ops.shape(delta_q)
    # Assumes keras_cached_hstu_mha is available
    return keras_cached_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )


def keras_hstu_preprocess_and_attention(
    x,
    norm_weight,
    norm_bias,
    norm_eps,
    num_heads,
    attn_dim,
    hidden_dim,
    uvqk_weight,
    uvqk_bias,
    max_seq_len,
    seq_offsets,
    attn_alpha,
    causal,
    num_targets,
    max_attn_len,
    contextual_seq_len,
    recompute_uvqk_in_backward,
    recompute_normed_x_in_backward,
    sort_by_length,
    prefill=False,
    kernel=None,
    **kwargs
) -> Tuple:
    u, q, k, v = hstu_compute_uqvk(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        norm_eps=norm_eps,
        num_heads=num_heads,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        uvqk_weight=uvqk_weight,
        uvqk_bias=uvqk_bias,
        kernel=kernel,
    )
    attn_output = keras_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=attn_alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        dropout_pr=0.0,
        training=False,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        sort_by_length=sort_by_length,
        kernel=kernel,
        **kwargs
    )
    attn_output = ops.reshape(attn_output, [-1, hidden_dim * num_heads])
    return u, attn_output, k, v


# ---- STU Layer ----


class STULayerConfig:
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention_dim: int,
        output_dropout_ratio: float = 0.3,
        causal: bool = True,
        target_aware: bool = True,
        max_attn_len: Optional[int] = None,
        attn_alpha: Optional[float] = None,
        use_group_norm: bool = False,
        recompute_normed_x: bool = True,
        recompute_uvqk: bool = True,
        recompute_y: bool = True,
        sort_by_length: bool = True,
        contextual_seq_len: int = 0,
    ):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.output_dropout_ratio = output_dropout_ratio
        self.causal = causal
        self.target_aware = target_aware
        self.max_attn_len = max_attn_len
        self.attn_alpha = attn_alpha
        self.use_group_norm = use_group_norm
        self.recompute_normed_x = recompute_normed_x
        self.recompute_uvqk = recompute_uvqk
        self.recompute_y = recompute_y
        self.sort_by_length = sort_by_length
        self.contextual_seq_len = contextual_seq_len


"""The _update_kv_cache function is responsible for managing the Key-Value (KV) cache
during both the full sequence prefill stage and the subsequent incremental generation steps.
It updates the cached history of k and v tensors
"""


def _update_kv_cache(
    max_seq_len,
    seq_offsets,
    k,
    v,
    max_kv_caching_len,
    kv_caching_lengths,
    orig_k_cache,
    orig_v_cache,
    orig_max_kv_caching_len,
    orig_kv_caching_offsets,
):
    if kv_caching_lengths is not None:
        kv_caching_offsets = ops.cast(
            ops.cumsum(kv_caching_lengths, exclusive=True), dtype="int32"
        )
        delta_offsets = seq_offsets - kv_caching_offsets
        k_values = ops.reshape(fx_unwrap_optional_tensor(k), [-1, ops.shape(k)[-1]])
        v_values = ops.reshape(fx_unwrap_optional_tensor(v), [-1, ops.shape(v)[-1]])
        k_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=k_values,
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        v_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=v_values,
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        if max_kv_caching_len == 0:
            max_kv_caching_len = ops.convert_to_numpy(
                ops.cast(ops.max(kv_caching_lengths), dtype="int32")
            ).item()
        return (k_cache, v_cache, max_kv_caching_len, kv_caching_offsets)
    else:
        return (
            orig_k_cache,
            orig_v_cache,
            orig_max_kv_caching_len,
            orig_kv_caching_offsets,
        )


"""The _construct_full_kv function is a crucial internal utility in the Structured
Transformer Unit (STU) responsible for combining the cached historical Key/Value vectors
with the newly computed vectors during the incremental inference phase. It generates the
full, continuous Key and Value tensors needed by the cached attention kernel.
"""


def _construct_full_kv(
    delta_k, delta_v, k_cache, v_cache, max_kv_caching_len, kv_caching_offsets
):
    L = ops.shape(delta_k)[0]
    B = ops.shape(kv_caching_offsets)[0] - 1
    delta_size = L // B
    full_k = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=k_cache,
        values_right=delta_k,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_v = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=v_cache,
        values_right=delta_v,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    delta_size_broadcast = delta_size * ops.arange(
        B + 1, dtype=kv_caching_offsets.dtype
    )
    full_kv_caching_offsets = kv_caching_offsets + delta_size_broadcast
    return (full_k, full_v, max_kv_caching_len + delta_size, full_kv_caching_offsets)


class STULayer(keras.layers.Layer):
    # A Keras layer implementing sequence-to-sequence attention with key-value caching for efficient inference.
    max_kv_caching_len: int = 0
    k_cache: Optional[keras.KerasTensor] = None
    v_cache: Optional[keras.KerasTensor] = None
    kv_caching_offsets: Optional[keras.KerasTensor] = None

    def __init__(self, config: STULayerConfig, is_inference: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        self._attn_alpha: float = config.attn_alpha or 1.0 / (self._attention_dim**0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len
        self.reset_kv_cache()

    def build(self, input_shape):
        D_in = input_shape[-1]
        H = self._num_heads
        A = self._attention_dim
        V = self._hidden_dim
        output_dim_total = (V * 2 + A * 2) * H
        self._uvqk_weight = self.add_weight(
            shape=(D_in, output_dim_total),
            initializer="glorot_uniform",
            name="uvqk_weight",
        )
        self._uvqk_beta = self.add_weight(
            shape=(output_dim_total,), initializer="zeros", name="uvqk_beta"
        )
        self._input_norm_weight = self.add_weight(
            shape=(D_in,), initializer="ones", name="input_norm_weight"
        )
        self._input_norm_bias = self.add_weight(
            shape=(D_in,), initializer="zeros", name="input_norm_bias"
        )
        self._output_weight = self.add_weight(
            shape=(V * H, self._embedding_dim),
            initializer="glorot_uniform",
            name="output_weight",
        )
        output_norm_shape: int = V * H if not self._use_group_norm else H
        self._output_norm_weight = self.add_weight(
            shape=(output_norm_shape,), initializer="ones", name="output_norm_weight"
        )
        self._output_norm_bias = self.add_weight(
            shape=(output_norm_shape,), initializer="zeros", name="output_norm_bias"
        )
        self.built = True

    def reset_kv_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.kv_caching_offsets = None
        self.max_kv_caching_len = 0

    def update_kv_cache(
        self, max_seq_len, seq_offsets, k, v, max_kv_caching_len, kv_caching_lengths
    ):
        self.k_cache, self.v_cache, self.max_kv_caching_len, self.kv_caching_offsets = (
            _update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                k=k,
                v=v,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                orig_k_cache=self.k_cache,
                orig_v_cache=self.v_cache,
                orig_max_kv_caching_len=self.max_kv_caching_len,
                orig_kv_caching_offsets=self.kv_caching_offsets,
            ),
        )

    def construct_full_kv(self, delta_k, delta_v):
        return _construct_full_kv(
            delta_k=delta_k,
            delta_v=delta_v,
            k_cache=fx_unwrap_optional_tensor(self.k_cache),
            v_cache=fx_unwrap_optional_tensor(self.v_cache),
            max_kv_caching_len=self.max_kv_caching_len,
            kv_caching_offsets=fx_unwrap_optional_tensor(self.kv_caching_offsets),
        )

    def call(
        self,
        x,
        x_lengths,
        x_offsets,
        max_seq_len,
        num_targets,
        max_kv_caching_len=0,
        kv_caching_lengths=None,
        training=None,
    ):
        u, attn_output, k, v = keras_hstu_preprocess_and_attention(
            x=x,
            norm_weight=self._input_norm_weight,
            norm_bias=self._input_norm_bias,
            norm_eps=1e-6,
            num_heads=self._num_heads,
            attn_dim=self._attention_dim,
            hidden_dim=self._hidden_dim,
            uvqk_weight=self._uvqk_weight,
            uvqk_bias=self._uvqk_beta,
            max_seq_len=max_seq_len,
            seq_offsets=x_offsets,
            attn_alpha=self._attn_alpha,
            causal=self._causal,
            num_targets=num_targets if self._target_aware else None,
            max_attn_len=self._max_attn_len,
            contextual_seq_len=self._contextual_seq_len,
            recompute_uvqk_in_backward=self._recompute_uvqk,
            recompute_normed_x_in_backward=self._recompute_normed_x,
            sort_by_length=self._sort_by_length,
            prefill=kv_caching_lengths is not None,
        )
        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=x_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )
        return hstu_compute_output(
            attn=attn_output,
            u=u,
            x=x,
            norm_weight=self._output_norm_weight,
            norm_bias=self._output_norm_bias,
            norm_eps=1e-6,
            dropout_ratio=self._output_dropout_ratio,
            output_weight=self._output_weight,
            group_norm=self._use_group_norm,
            num_heads=self._num_heads,
            linear_dim=self._hidden_dim,
            concat_ux=True,
            training=training,
            recompute_y_in_backward=self._recompute_y,
        )

    def cached_forward(
        self,
        delta_x,
        num_targets,
        max_kv_caching_len=0,
        kv_caching_lengths=None,
        training=None,
    ):
        A = self._attention_dim
        V = self._hidden_dim
        H = self._num_heads
        delta_u, delta_q, delta_k, delta_v = hstu_compute_uqvk(
            x=delta_x,
            norm_weight=self._input_norm_weight,
            norm_bias=self._input_norm_bias,
            norm_eps=1e-6,
            num_heads=self._num_heads,
            attn_dim=self._attention_dim,
            hidden_dim=self._hidden_dim,
            uvqk_weight=self._uvqk_weight,
            uvqk_bias=self._uvqk_beta,
        )
        k_flat = ops.reshape(delta_k, [-1, H * A])
        v_flat = ops.reshape(delta_v, [-1, H * V])
        k_full, v_full, max_seq_len, seq_offsets = self.construct_full_kv(
            delta_k=k_flat, delta_v=v_flat
        )
        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            k=k_full,
            v=v_full,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )
        k = ops.reshape(k_full, [-1, H, A])
        v = ops.reshape(v_full, [-1, H, V])
        delta_attn_output = delta_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=self._attn_alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets if self._target_aware else None,
            max_attn_len=self._max_attn_len,
            contextual_seq_len=self._contextual_seq_len,
        )
        delta_attn_output = ops.reshape(delta_attn_output, [-1, V * H])
        return hstu_compute_output(
            attn=delta_attn_output,
            u=delta_u,
            x=delta_x,
            norm_weight=self._output_norm_weight,
            norm_bias=self._output_norm_bias,
            norm_eps=1e-6,
            dropout_ratio=self._output_dropout_ratio,
            output_weight=self._output_weight,
            group_norm=self._use_group_norm,
            num_heads=self._num_heads,
            linear_dim=self._hidden_dim,
            concat_ux=True,
            training=training,
            recompute_y_in_backward=self._recompute_y,
        )


class STUStack(keras.layers.Layer):
    """
    A custom Keras layer that stacks multiple STULayer instances and applies them sequentially.

    Args:
        stu_layers (List[STULayer]): A list of STULayer instances to be applied in sequence.
        is_inference (bool, optional): Flag indicating whether the layer is used for inference. Defaults to False.
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Methods:
        call(x, x_lengths, x_offsets, max_seq_len, num_targets, max_kv_caching_len=0, kv_caching_lengths=None, training=None):
            Applies each STULayer in the stack sequentially to the input tensor `x`.
            Args:
                x: Input tensor.
                x_lengths: Lengths of the input sequences.
                x_offsets: Offsets for the input sequences.
                max_seq_len: Maximum sequence length.
                num_targets: Number of target outputs.
                max_kv_caching_len (int, optional): Maximum length for key-value caching. Defaults to 0.
                kv_caching_lengths (optional): Lengths for key-value caching. Defaults to None.
                training (optional): Training mode flag. Defaults to None.
            Returns:
                Tensor after sequentially applying all STULayers.

        cached_forward(delta_x, num_targets, max_kv_caching_len=0, kv_caching_lengths=None, training=None):
            Applies the cached_forward method of each STULayer in the stack sequentially.
            Args:
                delta_x: Input tensor for cached forward pass.
                num_targets: Number of target outputs.
                max_kv_caching_len (int, optional): Maximum length for key-value caching. Defaults to 0.
                kv_caching_lengths (optional): Lengths for key-value caching. Defaults to None.
                training (optional): Training mode flag. Defaults to None.
            Returns:
                Tensor after sequentially applying cached_forward of all STULayers.
    """

    def __init__(
        self, stu_layers: List[STULayer], is_inference: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self._stu_layers = stu_layers

    def call(
        self,
        x,
        x_lengths,
        x_offsets,
        max_seq_len,
        num_targets,
        max_kv_caching_len=0,
        kv_caching_lengths=None,
        training=None,
    ):
        for layer in self._stu_layers:
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                training=training,
            )
        return x

    def cached_forward(
        self,
        delta_x,
        num_targets,
        max_kv_caching_len=0,
        kv_caching_lengths=None,
        training=None,
    ):
        for layer in self._stu_layers:
            delta_x = layer.cached_forward(
                delta_x=delta_x,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                training=training,
            )
        return delta_x
