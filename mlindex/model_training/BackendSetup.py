"""Process-wide numerical settings for the torch backend.

These are global torch settings rather than per-model ones, so they live here and are called
explicitly from the training entry points. Importing a model class should not silently change the
precision every other matmul in the process runs at.
"""


def enable_tf32():
    """Let float32 matmuls use the Ampere tensor cores.

    torch ships with matmul TF32 off ('highest' precision), so every dense layer and attention
    matmul takes the plain fp32 path -- 19.5 TFLOPS on an A100 against 156 TFLOPS on the tensor
    cores. TF32 keeps float32's exponent and the leading 10 bits of mantissa and still accumulates
    in fp32, so it is a reduction in mantissa width, not in range: no rescaling, no loss scaling,
    and nothing to tune.

    Worth what the arithmetic intensity allows. Measured on the tetragonal integral filter, TF32
    alone is ~1.2x at batch 64, where the step is dominated by kernel launch overhead rather than
    matmul, rising to ~1.8x at batch 512 where it is genuinely compute bound.

    Harmless on hardware without tensor cores; torch ignores the flag there.
    """
    import torch
    torch.set_float32_matmul_precision('high')
