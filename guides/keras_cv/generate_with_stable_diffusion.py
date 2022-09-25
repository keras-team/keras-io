"""
Title: Generate images using KerasCV's StableDiffusion's at unprecedented speeds
Author: [divamgupta](https://github.com/divamgupta), [fchollet](https://github.com/fchollet), [lukewood](https://lukewood.xyz), [ianstenbit](https://github.com/ianstenbit), ... and others if anyone wants to contribute
Date created: 2022/09/24
Last modified: 2022/09/24
Description:
"""

"""
## Overview

In this guide, we will show how to generate novel images based on a text prompt using
the KerasCV implementation of [stability.ai's](https://stability.ai/) image to text
model,
[StableDiffusion](https://github.com/CompVis/stable-diffusion).

StableDiffusion is a powerful, open-source text to image generation model.  While there
exist numerous open source implementations that allow you to easily create images from
textual prompts, KerasCV's offers a few distinct advantages.
These include [XLA compilation](https://www.tensorflow.org/xla) and
[mixed precision computation](https://www.tensorflow.org/guide/mixed_precision).

In this guide, we will explore KerasCV's StableDiffusion implementation, show how to use
these powerful performance boosts, and explore the performance benefits
that they offer.

To get started, lets install a few dependencies and sort out some imports:
"""

"""shell
pip install ez-timer luketils
"""

import keras_cv
from luketils import visualization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from ez_timer import ez_timer

"""
## Introduction

Unlike most tutorials, where we first explain a topic then show how to implement it,
with text to image generation it is easiest to show instead of tell.

Check out the power of `keras_cv.models.StableDiffusion()`.
First, we construct a model:
"""

stable_diffusion = keras_cv.models.StableDiffusion()

"""
Next, we give it a prompt:
"""


def visualize_prompt(prompt, sd_model=None):
    sd_model = sd_model or stable_diffusion
    visualization.plot_gallery(
        sd_model.text_to_image(prompt, batch_size=3),
        rows=1,
        cols=3,
        scale=4,
        value_range=(0, 255),
        show=True,
    )


visualize_prompt("a cartoon caterpillar wearing glasses")

"""

Pretty incredible!

But that's not all this model can do.
Have you ever seen a monkey surf?
"""

visualize_prompt(
    "The worlds cutest monkey surfing on a " "beautiful wave in the sunshine!"
)

"""
Or the Garden of Eden filled with cats?
"""

visualize_prompt(
    "Absolutely stunning artwork of the garden of eden, " "but filled with cats!"
)

"""
Or a cute magical flying dog with a lengthy list of descriptors:
"""


visualize_prompt(
    "a cute magical flying dog, fantasy art drawn by disney concept artists, "
    "golden colour, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure"
)

"""
The possibilities are literally endless (or as you will learn later: at least extend to the boundaries of StableDiffusion's latent manifold).

Pretty incredible!  The idea should be self evident at this point.
Now lets take a step back and look at how this algorithm actually works.

## The StableDiffusion Algorithm

TODO(lukewood): write this
"""
# Need to write up the actual algorithm and provide an overview

"""
## Perks of KerasCV

So, with numerous implementations of StableDiffusion publicly available why shoud you use `keras_cv.models.StableDiffusion()`?

Aside from the easy-to-use API, KerasCV's StableDiffusion model comes with some nice bells and trinkets.  These extra features include but are not limited to:

- out of the box support for XLA compilation
- support for `mixed_precision`
- ...

When these are combined, the KerasCV StableDiffusion model runs orders of magnitude faster than naive implementations.  This section shows how to enable all of these features, and the resulting performance gain yielded from using them.

**Note: The difference between the performance benefits from each optimization vary drastically between hardware**
"""

"""
### XLA Compilation

TensorFlow comes with the [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) compiler built in.
`keras_cv.models.StableDiffusion` supports a `jit_compile` argument out of the box.
Setting this argument to `True` enables XLA compilation; resulting in a significant speed-up.

Lets use this below.
"""

xla_stable_diffusion = keras_cv.models.StableDiffusion(jit_compile=True)

# warm up the model by running inference once before timing it
visualize_prompt(
    "An oldschool macintosh computer showing an avocado on its screen",
    sd_model=xla_stable_diffusion,
)

"""
Now lets compare the runtimes of our models
"""

with ez_timer() as timer:
    visualize_prompt(
        "A cute water-colored otter in a rainbow whirlpool holding shells",
        sd_model=stable_diffusion,
    )
print(f"Without XLA took {timer.result} seconds")

"""
and now with XLA
"""

with ez_timer() as timer:
    visualize_prompt(
        "A cute water-colored otter in a rainbow whirlpool holding shells",
        sd_model=xla_stable_diffusion,
    )
print(f"With XLA took {timer.result} seconds")

"""
On my hardware I see about a 2x speedup.

### Mixed Precision

Mixed precision computation is the process of mixing `float32` and `float16` precision dtypes to take advantage of the fact that `float16` operations are significantly faster on modern accelarators.

While a low-level setting, enabling mixed precision computation in Keras (and therefore for `keras_cv.models.StableDiffusion`) is as simple as calling:

First, lets benchmark again without mixed precision:
"""

with ez_timer() as timer:
    visualize_prompt(
        "a cute magical flying dog, fantasy art drawn by disney concept artists, "
        "golden colour, high quality, highly detailed, elegant, sharp focus, "
        "concept art, character concepts, digital painting, mystery, adventure",
        sd_model=stable_diffusion,
    )
print(f"Without mixed precision took {timer.result} seconds")

"""
Now lets construct a model with mixed precision.
Here is what it takes to do so:
"""
mixed_precision.set_global_policy("mixed_float16")

"""
That's all.  Out of the box - it just works.
"""

stable_diffusion_mixed_precision = keras_cv.models.StableDiffusion()
print("Old compute dtype:", stable_diffusion.diffusion_model.compute_dtype)
print("Old variable dtype:", stable_diffusion.diffusion_model.variable_dtype)
print(
    "New compute dtype:", stable_diffusion_mixed_precision.diffusion_model.compute_dtype
)
print(
    "New variable dtype:",
    stable_diffusion_mixed_precision.diffusion_model.variable_dtype,
)

"""
As you can see, the model constructed above now uses mixed precision computation;
leveraging the speed of `float16` for computation, and `float32` to store variables.
"""
# warm up model to run graph tracing before benchmarking
stable_diffusion_mixed_precision.text_to_image("warming up the model")

with ez_timer() as timer:
    visualize_prompt(
        "a cute magical flying dog, fantasy art drawn by disney concept artists, "
        "golden colour, high quality, highly detailed, elegant, sharp focus, "
        "concept art, character concepts, digital painting, mystery, adventure",
        sd_model=stable_diffusion_mixed_precision,
    )
print(f"With mixed precision took {timer.result} seconds")

"""
## Putting It All Together

So?  How do you assemble the world's most performant stable diffusion inference pipeline (as of September 2022).

Two lines of code:
"""
mixed_precision.set_global_policy("mixed_float16")
supermodel = keras_cv.models.StableDiffusion(jit_compile=True)
"""
and to use it...
"""
supermodel.text_to_image("warming up the model")

"""
Exactly how fast is it?
Lets find out!
"""

with ez_timer() as timer:
    visualize_prompt(
        "A mysterious dark stranger visits the great pyramids of egypt, "
        "high quality, highly detailed, elegant, sharp focus, "
        "concept art, character concepts, digital painting",
        sd_model=supermodel,
    )
print(f"With XLA and mixed precision took {timer.result} seconds")

"""
Four seconds to generate three novel images from a text prompt.

What a time to be alive!
"""

"""
## Engineering Good Prompts

You may notice above that some of the prompts have descriptors such as:
```
"high quality, highly detailed, elegant, sharp focus, "
"concept art, character concepts, digital painting",
```

Why is this?

TODO(ianstenbit): Some help would be great
"""

"""
## Benchmarks

@ianstenbit can write this
"""

"""
## Conclusions

KerasCV offers a high quality API to leverage StableDiffusion today.
Through the use of XLA and mixed precision Tensorflow allows us to construct the fastest StableDiffusion pipeline available as of September 2022.

Normally, at the end of a keras.io tutorial we leave you with some future directions to continue in to learn.
This time, we leave you with one idea:

**Go run your own prompts through the model!  It is an absolute blast!**
"""
