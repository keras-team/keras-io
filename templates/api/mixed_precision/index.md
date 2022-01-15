# Mixed precision

## What is mixed precision training?

Mixed precision training is the use of lower-precision operations (`float16` and `bfloat16`) in a model
during training to make it run faster and use less memory.
Using mixed precision can improve performance by more than 3 times on modern GPUs and 60% on TPUs.

Today, most models use the `float32` dtype, which takes 32 bits of memory.
However, there are two lower-precision dtypes, `float16` and `bfloat16`,
each which take 16 bits of memory instead. Modern accelerators like Google TPUs and NVIDIA GPUs 
can run operations faster in the 16-bit dtypes,
as they have specialized hardware to run 16-bit computations and 16-bit dtypes can be read from memory faster.
Therefore, these lower-precision dtypes should be used whenever possible on those devices.

However, variables storage (as well as certain sensitive computations) should still be in `float32`
to preserve numerical stability. By using 16-bit precision whenever possible and keeping certain critical
parts of the model in `float32`, the model will run faster,
while training as well as when using 32-bit precision.


## Using mixed precision training in Keras

The precision policy used by Keras layers or models is controled by a `tf.keras.mixed_precision.Policy` instance.
Each layer has its own `Policy`. You can either set it on an individual layer via the `dtype` argument
(e.g. `MyLayer(..., dtype="mixed_float16")`), or you can set a global value to be used by all layers by
default, via the utility `tf.keras.mixed_precision.set_global_policy`.

Typically, to start using mixed precision on GPU, you would simply call `tf.keras.mixed_precision.set_global_policy("mixed_float16")`
at the start of your program. On TPU, you would call `tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")`.


## API documentation

{{toc}}


## Supported hardware

While mixed precision will run on most hardware, it will only speed up models on recent NVIDIA GPUs and Google TPUs.
NVIDIA GPUs support using a mix of float16 and float32, while TPUs support a mix of bfloat16 and float32.

Among NVIDIA GPUs, those with compute capability 7.0 or higher will see the greatest performance benefit
from mixed precision because they have special hardware units, called Tensor Cores,
to accelerate float16 matrix multiplications and convolutions. Older GPUs offer no math
performance benefit for using mixed precision, however memory and bandwidth savings can enable some speedups.
You can look up the compute capability for your GPU at NVIDIA's [CUDA GPU web page](https://developer.nvidia.com/cuda-gpus).
Examples of GPUs that will benefit most from mixed precision include RTX GPUs, the V100, and the A100.

Even on CPUs and older GPUs, where no speedup is expected, mixed precision APIs can still be used for unit testing,
debugging, or just to try out the API. On CPUs, mixed precision will run significantly slower, however.


You can check your GPU type with the following command:

```
nvidia-smi -L
```