from enum import Enum, IntEnum

import torch
from torchao.prototype.custom_fp_utils import (
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
)


FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1


def down_size(size):
    if size[-1] % 2 != 0:
        raise ValueError(f"{size} last dim not divisible by two")
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def unpack_uint4(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    if not uint8_data.is_contiguous():
        raise RuntimeError("uint8_data is not contiguous in memory")

    shape = uint8_data.shape

    # since we are using uint8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    first_elements = (uint8_data >> 4).to(torch.uint8)
    second_elements = (uint8_data & 0b1111).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
        up_size(shape)
    )

    return unpacked


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    if shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension of shape {shape} must be divisible by 2")
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size(shape))


def f32_to_f4_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-3 empty and
      bits 4-7 in fp4_e2m1
    """
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def f4_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-3 empty and bits 4-7
      containing an fp4_e2m1 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _floatx_unpacked_to_f32(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


# Enum for rounding modes
class RoundingMode(IntEnum):
    nearest = 0
    floor = 1
    even = 2

    @staticmethod
    def string_enums():
        return [s.name for s in list(RoundingMode)]


def round_to_decimal(x):
    abs_x = torch.abs(x)
    exponent = torch.floor(torch.log2(abs_x))
    mantissa = abs_x / (2**exponent)
    exponent = torch.where(mantissa > 1.75, exponent + 1, exponent)
    return exponent


def _shared_exponents(A, method="max", axes=None, ebits=0, shared_exp_round_method="floor"):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    shared_exp = shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
    if shared_exp_round_method == "floor":
        shared_exp = torch.floor(torch.log2(shared_exp))
    elif shared_exp_round_method == "round":
        shared_exp = torch.log2(shared_exp).round()
    elif shared_exp_round_method == "round2decimal":
        shared_exp = round_to_decimal(shared_exp)
    else:
        raise Exception("Unrecognized round method!")

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits - 1) - 1
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    if not all(x >= 0 for x in axes):
        raise ValueError(f"All axes must be non-negative, got {axes}")
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                if shape[axis] % reshape_block_size != 0:
                    raise ValueError(
                        f"Dimension at axis {axis} (value {shape[axis]}) must be divisible by reshape_block_size {reshape_block_size}")
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    A = A.view(padded_shape)
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A


def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)



def _get_min_norm(ebits):
    """ Valid for all float formats """
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin


def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round='nearest',
                            saturate_normals=False, allow_denorm=True,
                            custom_cuda=False):
    """ Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")

        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2**(exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    # handle Inf/NaN
    out[A == float("Inf")] = float("Inf")
    out[A == -float("Inf")] = -float("Inf")
    out[A == float("NaN")] = float("NaN")

    return out


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def quantize_mx(
    A,
    quant_bit,
    scale_bits=8,
    fp32_scale=False,
    shared_exp_method="max",
    axes=-1,
    block_size=32,
    round="nearest",
    flush_fp32_subnorms=False,
    shared_exp_round_method="round2decimal",
    real_quant=False,
):
    """Function used for MX* quantization
    """

    if not (scale_bits > 0):
        raise ValueError(f"scale_bits must be greater than 0, got {scale_bits}")

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    if quant_bit == 8:
        ebits, mbits, emax, max_norm = 4, 5, 8, 448.0 # e4m3
    elif quant_bit == 4:
        ebits, mbits, emax, max_norm = 2, 3, 2, 6.0 # e2m1
    else:
        raise Exception("quant_bit must be 8 or 4")

    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    if fp32_scale:
        for axis in axes:
            shared_exp, _ = torch.max(torch.abs(A), dim=axis, keepdim=True)
        scale = shared_exp / max_norm
        A = A / scale
    else:
        shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

        # Get shared exponents
        shared_exp = _shared_exponents(
            A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
            shared_exp_round_method=shared_exp_round_method
        )

        # Flush subnormal FP32 inputs to zero
        if flush_fp32_subnorms:
            A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - emax

        scale_emax = 2**(scale_bits - 1) - 1
        shared_exp[shared_exp > scale_emax] = float("NaN")
        shared_exp[shared_exp < -scale_emax] = -scale_emax

        A = A / (2**shared_exp)

    A = _quantize_elemwise_core(
            A, mbits, ebits, max_norm, round=round,
            allow_denorm=True, saturate_normals=True,
            custom_cuda=False)
    if real_quant:
        if block_size:
            A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)
        return A.to(torch.float8_e4m3fn), (shared_exp + 127).to(torch.uint8).squeeze(-1)
    if fp32_scale:
        A = A * scale
    else:
        A = A * (2**shared_exp)

    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A, shared_exp