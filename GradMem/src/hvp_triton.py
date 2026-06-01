"""
Fused Attention
======================================================================

This is an extension of the Triton kernel written by the original authors of the GradMem paper, shared here:
https://github.com/yurakuratov/gradmem/blob/main/attn_double_bwd/hvp_semi_manual.py


Below is from the original frontmatter written by the GradMem authors:
======================================================================


This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team
Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

Plus modifications to support Jacobian-vector products (JVPs) and Hessian-vector products (HVPs):
- Formulation of flash JVP, by Cheng Lu and Yang Song in https://arxiv.org/abs/2410.11081.
- Reference Triton implementation, by Sofian Mejjoute.
- Reimplementing reference implementation as an autograd function with latest Triton tutorial optimizations, by Alex Birch.
- Support for forward to receive tangents, so as to compute fwd and jvp together; autograd workaround, by Emily (nshepperd).
- Support for function transforms (e.g., torch.func.jvp) via the use of setup_context, by Shih-Ying Yeh.
- Support for sequence lengths 32 & 64; float32 & bfloat16 precision; comprehensive, length and dtype-stratified unit tests;
    working backward hook w.r.t. tensor contiguity; HVP stress testing; standardized docstrings/packaging; and masking/dropout, by Alex Morehead.
"""

from __future__ import annotations

import os
from typing import Any, Literal, NamedTuple, Optional

import torch
import torch.autograd.forward_ad as fwAD
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from transformers.integrations.sdpa_attention import repeat_kv

# NOTE: Uncomment to turn warnings into errors for debugging
# import warnings
# warnings.filterwarnings("error", category=UserWarning)
# warnings.filterwarnings("error", category=RuntimeWarning)

try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    HAS_TENSOR_DESC = True
except ModuleNotFoundError:
    HAS_TENSOR_DESC = False

MASK_CONST = (
    -1.0e2
)  # Use a large negative value for masking (compatible with float16, bfloat16, and float32)
MIN_SEQUENCE_LENGTH = 32  # NOTE: All sequence lengths must be multiples of 2 >= 32


def is_hip():
    """Check if the current device is HIP."""
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


def is_cuda():
    """Check if the current device is CUDA."""
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except Exception:
        return False


def supports_host_descriptor():
    """Check if the current device supports host tensor descriptors."""
    try:
        return is_cuda() and torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


def supports_tma():
    """Check if the current device supports Tensor Memory Access (TMA)."""
    try:
        return HAS_TENSOR_DESC and is_cuda() and torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


def is_blackwell():
    """Check if the current device is Blackwell architecture."""
    try:
        return is_cuda() and torch.cuda.get_device_capability()[0] == 10
    except Exception:
        return False


@triton.jit
def create_dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    """Generate dropout mask using Philox RNG.

    Args:
        philox_seed: Seed for Philox RNG.
        philox_offset: Offset for Philox RNG.
        dropout_p: Dropout probability.
        m: Number of rows.
        n: Number of columns.
        stride: Stride for the output mask.

    Returns:
        dropout_mask: A boolean mask indicating which elements to keep (1.0) or drop (0.0).
        dropout_scale: Scale factor to apply after dropout.
    """
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    offs = ms[:, None] * stride + ns[None, :]
    rng_offs = philox_offset + offs

    # Generate random values using Philox
    rand_vals = tl.rand(philox_seed, rng_offs)

    # Create dropout mask (1.0 = keep, 0.0 = drop)
    dropout_mask = rand_vals > dropout_p
    dropout_scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0

    return dropout_mask, dropout_scale


@triton.jit
def _attn_fwd_inner(
    acc,
    g_acc,  #
    l_i,
    m_i,  #
    mu_i,
    p_tv_acc,  #
    q,
    t_q,  #
    K_block_ptr,
    V_block_ptr,  #
    T_K_block_ptr,
    T_V_block_ptr,  #
    # Mask and dropout parameters
    mask_block_ptr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    # Other parameters
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    sm_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,  #
    ENABLE_JVP: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_TYPE: tl.constexpr,  # 0: no mask, 1: boolean, 2: additive
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    """Inner forward pass for attention mechanism.

    Args:
        acc: Accumulator tensor.
        g_acc: Gradient accumulator tensor.
        l_i: Tensor for storing intermediate results.
        m_i: Tensor for storing intermediate results.
        mu_i: Tensor for storing intermediate results.
        p_tv_acc: Tensor for storing intermediate results.
        q: Query tensor.
        t_q: Tangent of the query tensor.
        K_block_ptr: Pointer to the key block.
        V_block_ptr: Pointer to the value block.
        T_K_block_ptr: Pointer to the tangent key block.
        T_V_block_ptr: Pointer to the tangent value block.
        mask_block_ptr: Pointer to the attention mask block.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        dtype: Data type of the tensors.
        start_m: Starting index for the current block.
        qk_scale: Scale factor for the query-key dot product.
        sm_scale: Scale factor for the softmax.
        BLOCK_M: Block size for the M dimension.
        HEAD_DIM: Dimension of the attention heads.
        BLOCK_N: Block size for the N dimension.
        STAGE: Current stage of the computation.
        offs_m: Offsets for the M dimension.
        offs_n: Offsets for the N dimension.
        N_CTX: Number of context tokens.
        warp_specialize: Whether to apply warp specialization.
        ENABLE_JVP: Whether to enable JVP (Jacobian-vector product).
        ENABLE_DROPOUT: Whether to enable dropout.
        MASK_TYPE: Type of attention mask (0: no mask, 1: boolean, 2: additive).
        MASK_CONST: Constant value used for masking.

    Returns:
        The output tensors as a tuple.
    """
    # Range of values handled by this stage
    if STAGE == 1:
        # NOTE: From 0 to the left of the diagonal
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # NOTE: Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # NOTE: Only used for non-causal attention
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # NOTE: In fp8 mode, we may want to advance the V_block_ptr differently.
    # I did try advancing by (0, lo) instead for fp8, but I got an illegal memory access.
    # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    if ENABLE_JVP:
        T_K_block_ptr = tl.advance(T_K_block_ptr, (0, lo))
        T_V_block_ptr = tl.advance(T_V_block_ptr, (lo, 0))

    if MASK_TYPE > 0:
        mask_block_ptr = tl.advance(mask_block_ptr, (0, lo))

    # Loop over k, v and update accumulator
    # Use tl.range with warp_specialize on Hopper for producer/consumer decoupling
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- Compute qk --
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if ENABLE_JVP:
            t_k = tl.load(T_K_block_ptr)
            t_qk = tl.dot(t_q, k) + tl.dot(q, t_k)

        # Load and apply attention mask if provided (before scaling for STAGE != 2)
        if MASK_TYPE > 0:
            mask = tl.load(mask_block_ptr)
            if MASK_TYPE == 1:  # Boolean mask
                # Convert boolean to additive mask: True (attend) -> 0, False (ignore) -> -inf
                qk = qk + tl.where(mask == 1, 0.0, MASK_CONST)
                if ENABLE_JVP:
                    t_qk = tl.where(mask == 1, t_qk, 0.0)

            elif MASK_TYPE == 2:  # Additive mask
                qk = qk + mask

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # For causal attention (STAGE == 2)
        elif STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, MASK_CONST)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        # No masking case (MASK_TYPE == 0 and STAGE != 2)
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)

        if MASK_TYPE == 1 or STAGE == 2:
            # Account for fully masked sequence blocks
            p = tl.where(mask == 1, p, 0.0)

        # Apply dropout if enabled
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + (start_m * BLOCK_M) * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX
            )
            p = p * dropout_mask.to(dtype) * dropout_scale

        l_ij = tl.sum(p, 1)

        # -- Update m_i and l_i --
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # -- Update output accumulator --
        if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        # NOTE: We may need to transpose v if dtype == tl.float8e5
        # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
        p = p.to(dtype)

        if ENABLE_JVP:
            p_tqk = p * (t_qk * sm_scale)

            if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
                BM: tl.constexpr = g_acc.shape[0]
                BN: tl.constexpr = g_acc.shape[1]
                g_acc0, g_acc1 = g_acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                g_acc0 = g_acc0 * alpha[:, None]
                g_acc1 = g_acc1 * alpha[:, None]
                g_acc = tl.join(g_acc0, g_acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                g_acc = g_acc * alpha[:, None]

            g_acc = tl.dot(p_tqk.to(v.dtype), v, g_acc)
            mu_ij = tl.sum(p_tqk, 1)
            mu_i = mu_i * alpha + mu_ij
            t_v = tl.load(T_V_block_ptr)
            p_tv_acc = p_tv_acc * alpha[:, None] + tl.dot(p, t_v.to(dtype)).to(t_v.dtype)
            T_V_block_ptr = tl.advance(T_V_block_ptr, (BLOCK_N, 0))
            T_K_block_ptr = tl.advance(T_K_block_ptr, (0, BLOCK_N))

        acc = tl.dot(p, v.to(dtype), acc).to(acc.dtype)

        # -- Update m_i --
        m_i = m_ij

        # -- Move to the next block of K, V, and maybe the mask --
        # NOTE: The fp8 PR made a change to how K and V are advanced here but I believe we already have that.
        # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

        if MASK_TYPE > 0:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))

    return acc, g_acc, l_i, m_i, mu_i, p_tv_acc


@triton.jit
def _attn_fwd_inner_tma(
    acc,
    g_acc,  #
    l_i,
    m_i,  #
    mu_i,
    p_tv_acc,  #
    q,
    t_q,  #
    desc_k,
    desc_v,  #
    desc_k_t,
    desc_v_t,  #
    offset_y,
    # Mask and dropout parameters
    mask_block_ptr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    # Other parameters
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    sm_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    ENABLE_JVP: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_TYPE: tl.constexpr,  # 0: no mask, 1: boolean, 2: additive
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    """Inner forward pass for attention mechanism with TMA (Tensor Memory Access) support.

    Args:
        acc: Accumulator tensor.
        g_acc: Gradient accumulator tensor.
        l_i: Tensor for layer normalization.
        m_i: Tensor for masking.
        mu_i: Tensor for mean.
        p_tv_acc: Tensor for TV attention.
        q: Query tensor.
        t_q: Transposed query tensor.
        desc_k: Descriptor for key tensor.
        desc_v: Descriptor for value tensor.
        desc_k_t: Descriptor for transposed key tensor.
        desc_v_t: Descriptor for transposed value tensor.
        offset_y: Offset for y dimension.
        mask_block_ptr: Pointer to the attention mask block.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        dtype: Data type.
        start_m: Start index for m dimension.
        qk_scale: Scale factor for qk.
        sm_scale: Scale factor for sm.
        BLOCK_M: Block size for m dimension.
        HEAD_DIM: Head dimension size.
        BLOCK_N: Block size for n dimension.
        STAGE: Stage of computation.
        offs_m: Offset for m dimension.
        offs_n: Offset for n dimension.
        N_CTX: Context size.
        warp_specialize: Flag for warp specialization.
        ENABLE_JVP: Flag for enabling JVP.
        ENABLE_DROPOUT: Flag for enabling dropout.
        MASK_TYPE: Type of attention mask (0: no mask, 1: boolean, 2: additive).
        MASK_CONST: Constant value used for masking.

    Returns:
        The output tensors as a tuple.
    """
    # Range of values handled by this stage
    if STAGE == 1:
        # NOTE: From 0 to the left of the diagonal
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # NOTE: Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # NOTE: Only used for non-causal attention
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    if MASK_TYPE > 0:
        mask_block_ptr = tl.advance(mask_block_ptr, (0, lo))

    # Loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        # Let the compiler know that start_n is a multiple
        # of BLOCK_N, so the compiler can do optimizations
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- Compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if ENABLE_JVP:
            t_k = desc_k_t.load([offsetk_y, 0]).T
            t_qk = tl.dot(t_q, k) + tl.dot(q, t_k)

        # Load and apply attention mask if provided (before scaling for STAGE != 2)
        if MASK_TYPE > 0:
            mask = tl.load(mask_block_ptr)
            if MASK_TYPE == 1:  # Boolean mask
                # Convert boolean to additive mask: True (attend) -> 0, False (ignore) -> -inf
                qk = qk + tl.where(mask == 1, 0.0, MASK_CONST)
                if ENABLE_JVP:
                    t_qk = tl.where(mask == 1, t_qk, 0.0)

            elif MASK_TYPE == 2:  # Additive mask
                qk = qk + mask

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # For causal attention (STAGE == 2)
        elif STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, MASK_CONST)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        # No masking case (MASK_TYPE == 0 and STAGE != 2)
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)

        if MASK_TYPE == 1 or STAGE == 2:
            # Account for fully masked sequence blocks
            p = tl.where(mask == 1, p, 0.0)

        # Apply dropout if enabled
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + (start_m * BLOCK_M) * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX
            )
            p = p * dropout_mask.to(dtype) * dropout_scale

        # -- Compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # -- Update output accumulator --
        if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]

        # Prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])

        p = p.to(dtype)

        if ENABLE_JVP:
            p_tqk = p * (t_qk * sm_scale)

            # NOT: This non-transposed v for FP8 is presumably only supported on Blackwell
            if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
                BM: tl.constexpr = g_acc.shape[0]
                BN: tl.constexpr = g_acc.shape[1]
                g_acc0, g_acc1 = g_acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                g_acc0 = g_acc0 * alpha[:, None]
                g_acc1 = g_acc1 * alpha[:, None]
                g_acc = tl.join(g_acc0, g_acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                g_acc = g_acc * alpha[:, None]

            g_acc = tl.dot(p_tqk.to(v.dtype), v, g_acc)
            mu_ij = tl.sum(p_tqk, 1)
            mu_i = mu_i * alpha + mu_ij
            t_v = desc_v_t.load([offsetv_y, 0])
            p_tv_acc = p_tv_acc * alpha[:, None] + tl.dot(p, t_v.to(dtype)).to(t_v.dtype)

        # NOTE: This non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v.to(dtype), acc).to(acc.dtype)

        # Update m_i and l_i
        # Place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

        if MASK_TYPE > 0:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))

    return acc, g_acc, l_i, m_i, mu_i, p_tv_acc


def _host_descriptor_pre_hook(nargs):
    """Pre-hook to set up tensor descriptors for the attention kernel.

    Args:
        nargs: A dictionary of kernel arguments.
    """
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not supports_tma() or not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s,
        num_warps=w,
        pre_hook=_host_descriptor_pre_hook,
    )
    for BM in [MIN_SEQUENCE_LENGTH, 64, 128]
    for BN in [MIN_SEQUENCE_LENGTH, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(
            dict(BLOCK_M=128, BLOCK_N=64),
            num_stages=2,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        ),
    ]


def keep(conf):
    """Keep configurations that meet certain criteria.

    Args:
        conf: A configuration object.
    """
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    """Prune configurations that are invalid based on certain criteria.

    Args:
        configs: A list of configuration objects.
        named_args: A dictionary of named arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        A list of valid configuration objects.
    """
    N_CTX = kwargs["N_CTX"]

    if N_CTX == MIN_SEQUENCE_LENGTH:
        # Filter out configs where BLOCK_M > MIN_SEQUENCE_LENGTH
        return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= MIN_SEQUENCE_LENGTH]

    # Filter out configs where BLOCK_M > N_CTX or BLOCK_M <= MIN_SEQUENCE_LENGTH, as
    # BLOCK_M = MIN_SEQUENCE_LENGTH often leads to reduced numerical accuracy for longer sequences
    # TODO: Find out why this occurs
    return [
        conf for conf in configs if MIN_SEQUENCE_LENGTH < conf.kwargs.get("BLOCK_M", 0) <= N_CTX
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    """Maybe make a tensor descriptor from a pointer.

    Args:
        desc_or_ptr: The input tensor or pointer.
        shape: The shape of the tensor.
        strides: The strides of the tensor.
        block_shape: The block shape of the tensor.

    Returns:
        A tensor descriptor.
    """
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# @triton.autotune(
#     configs=list(filter(keep, configs)),
#     key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
#     prune_configs_by={"early_config_prune": prune_invalid_configs},
# )
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    T_Q,
    T_K,
    T_V,  #
    sm_scale,
    M,
    Out,
    T_Out,  #
    Mask,  # Mask tensor
    dropout_p,  # Dropout probability
    philox_seed,  # RNG seed for dropout
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_tqz,
    stride_tqh,
    stride_tqm,
    stride_tqk,  #
    stride_tkz,
    stride_tkh,
    stride_tkn,
    stride_tkk,  #
    stride_tvz,
    stride_tvh,
    stride_tvk,
    stride_tvn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    stride_toz,
    stride_toh,
    stride_tom,
    stride_ton,  #
    stride_mz,  # Mask stride
    stride_mh,  # Mask stride
    stride_mm,  # Mask stride
    stride_mn,  # Mask stride
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    ENABLE_JVP: tl.constexpr,  #
    ENABLE_DROPOUT: tl.constexpr,  #
    MASK_TYPE: tl.constexpr,  #
):
    """Forward attention computation.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        T_Q: Tensor for query.
        T_K: Tensor for key.
        T_V: Tensor for value.
        sm_scale: Scale factor.
        M: Number of rows.
        Out: Output tensor.
        T_Out: Tensor for output.
        Mask: Attention mask tensor.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        stride_qz: Stride for query z dimension.
        stride_qh: Stride for query h dimension.
        stride_qm: Stride for query m dimension.
        stride_qk: Stride for query k dimension.
        stride_kz: Stride for key z dimension.
        stride_kh: Stride for key h dimension.
        stride_kn: Stride for key n dimension.
        stride_kk: Stride for key k dimension.
        stride_vz: Stride for value z dimension.
        stride_vh: Stride for value h dimension.
        stride_vk: Stride for value k dimension.
        stride_vn: Stride for value n dimension.
        stride_tqz: Stride for tensor query z dimension.
        stride_tqh: Stride for tensor query h dimension.
        stride_tqm: Stride for tensor query m dimension.
        stride_tqk: Stride for tensor query k dimension.
        stride_tkz: Stride for tensor key z dimension.
        stride_tkh: Stride for tensor key h dimension.
        stride_tkn: Stride for tensor key n dimension.
        stride_tkk: Stride for tensor key k dimension.
        stride_tvz: Stride for tensor value z dimension.
        stride_tvh: Stride for tensor value h dimension.
        stride_tvk: Stride for tensor value k dimension.
        stride_tvn: Stride for tensor value n dimension.
        stride_oz: Stride for output z dimension.
        stride_oh: Stride for output h dimension.
        stride_om: Stride for output m dimension.
        stride_on: Stride for output n dimension.
        stride_toz: Stride for tensor output z dimension.
        stride_toh: Stride for tensor output h dimension.
        stride_tom: Stride for tensor output m dimension.
        stride_ton: Stride for tensor output n dimension.
        stride_mz: Stride for mask z dimension.
        stride_mh: Stride for mask h dimension.
        stride_mm: Stride for mask m dimension.
        stride_mn: Stride for mask n dimension.
        Z: Number of z dimensions.
        H: Number of h dimensions.
        N_CTX: Number of context dimensions.
        HEAD_DIM: Head dimension.
        BLOCK_M: Block size for the queries.
        BLOCK_N: Block size for the keys/values.
        FP8_OUTPUT: FP8 output flag.
        STAGE: Stage.
        warp_specialize: Warp specialization flag.
        ENABLE_JVP: Enable JVP flag.
        ENABLE_DROPOUT: Enable dropout flag.
        MASK_TYPE: Mask type (0: no mask, 1: boolean, 2: additive).
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)  # N = KV

    # Prepare metadata and indices
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float32  # For dot products
    start_m = tl.program_id(0)  # Which block (in the input query sequence) to process
    off_hz = tl.program_id(
        1
    )  # Which head and batch element to process, with a program being a single head of a single batch element
    off_z = (
        off_hz // H
    )  # Which batch element this program is assigned to (n.b., each batch element has H heads)
    off_h = off_hz % H  # The position of the head to process in the batch

    # NOTE: This allows one to get the (N_CTX, HEAD_DIM) block in Q, K, V by indexing it by batch and head
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Initialize block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),  # M = Q
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(
            stride_kk,
            stride_kn,
        ),  # NOTE: We invert the strides of K to get its matrix transpose K^T
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # Initialize block pointer for the mask, if provided
    if MASK_TYPE > 0:
        mask_offset = off_z.to(tl.int64) * stride_mz + off_h.to(tl.int64) * stride_mh
        mask_block_ptr = tl.make_block_ptr(
            base=Mask + mask_offset,
            shape=(N_CTX, N_CTX),
            strides=(stride_mm, stride_mn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        mask_block_ptr = None

    # Initialize dropout offset for this block
    philox_offset_base = off_hz * N_CTX * N_CTX

    # Initialize offsets for the query tokens to process
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulator pointers:
    # m, the running maximum (one for each query)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l, the running sum (one for each query as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc, the output accumulator (one vector for each query)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if ENABLE_JVP:
        # NOTE: It's extremely likely we could just reuse qvk_offset, but this seems cheap so whatever
        t_qvk_offset = off_z.to(tl.int64) * stride_tqz + off_h.to(tl.int64) * stride_tqh
        T_Q_block_ptr = tl.make_block_ptr(
            base=T_Q + t_qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_tqm, stride_tqk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # NOTE: Could probably just reuse v_order here
        t_v_order: tl.constexpr = (0, 1) if T_V.dtype.element_ty == tl.float8e5 else (1, 0)
        T_V_block_ptr = tl.make_block_ptr(
            base=T_V + t_qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_tvk, stride_tvn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=t_v_order,
        )
        T_K_block_ptr = tl.make_block_ptr(
            base=T_K + t_qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(
                stride_tkk,
                stride_tkn,
            ),  # NOTE: We invert the strides of tangent K (k_t) to get its matrix transpose K^T
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        T_O_block_ptr = tl.make_block_ptr(
            base=T_Out + t_qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_tom, stride_ton),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # Load q_t: It will stay in SRAM throughout.
        t_q = tl.load(T_Q_block_ptr)
        g_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        mu_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        p_tv_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    else:
        t_q = None
        T_V_block_ptr = None
        T_K_block_ptr = None
        # Allocate minimal dummy tensors to keep consistent the return signature of _attn_fwd_inner
        g_acc = tl.zeros([1, 1], dtype=tl.float32)
        mu_i = tl.zeros([1], dtype=tl.float32)
        p_tv_acc = tl.zeros([1, 1], dtype=tl.float32)

    # Prepare scales
    qk_scale = sm_scale
    qk_scale = qk_scale * 1.44269504  # 1/log(2)

    # Load q: It will stay in SRAM throughout.
    q = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1
    if STAGE == 1 or STAGE == 3:
        # NOTE: This step runs for non-causal attention or for the
        # blocks to the left of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner(
            acc,
            g_acc,
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            K_block_ptr,
            V_block_ptr,  #
            T_K_block_ptr,
            T_V_block_ptr,  #
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    if STAGE == 3:
        # NOTE: This step runs for the blocks to the
        # right of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner(
            acc,
            g_acc,  #
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            K_block_ptr,
            V_block_ptr,  #
            T_K_block_ptr,
            T_V_block_ptr,  #
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    # Epilogue
    empty_mask = l_i == 0.0
    # Unconditionally apply the mask to avoid data-dependent branching
    l_i = tl.where(empty_mask, 1.0, l_i)

    m_i = m_i + tl.where(
        # NOTE: This is needed to compute the logsumexp for the backward pass.
        empty_mask,
        0.0,
        tl.math.log2(l_i),
    )

    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

    # If JVP is enabled, compute and store the output tangent
    if ENABLE_JVP:
        t_p_v = g_acc / l_i[:, None] - (mu_i / l_i)[:, None] * acc
        t_y_out = t_p_v + p_tv_acc / l_i[:, None]
        tl.store(T_O_block_ptr, t_y_out.to(T_Out.type.element_ty))


def _tma_pre_hook(nargs):
    """Pre-hook for TMA (Tensor Memory Access) optimization.

    Args:
        nargs: A dictionary containing the kernel arguments.
    """
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs_tma = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s,
        num_warps=w,
        pre_hook=_tma_pre_hook,
    )
    for BM in [MIN_SEQUENCE_LENGTH, 64, 128, 256]
    for BN in [MIN_SEQUENCE_LENGTH, 64, 128]
    for s in [3, 4, 5]
    for w in [4, 8]
]


def keep_tma(conf):
    """Check if TMA (Tensor Memory Access) optimization should be kept for the given configuration.

    Args:
        conf: The configuration to check.
    """
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


# @triton.autotune(
#     configs=list(filter(keep_tma, configs_tma)),
#     key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
#     prune_configs_by={"early_config_prune": prune_invalid_configs},
# )
@triton.jit
def _attn_fwd_tma(
    sm_scale,
    M,  #
    Z,
    H,  #
    desc_q,
    desc_k,
    desc_v,  #
    desc_q_t,
    desc_k_t,
    desc_v_t,  #
    desc_o,
    desc_o_t,  #
    Mask,  # Mask tensor
    dropout_p,  # Dropout probability
    philox_seed,  # RNG seed for dropout
    stride_mz,  # Mask stride
    stride_mh,  # Mask stride
    stride_mm,  # Mask stride
    stride_mn,  # Mask stride
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    ENABLE_JVP: tl.constexpr,  #
    ENABLE_DROPOUT: tl.constexpr,  #
    MASK_TYPE: tl.constexpr,  #
):
    """Forward attention computation with TMA (Tensor Memory Access) support.

    Args:
        sm_scale: Scale factor for the softmax.
        M: Number of rows in the input.
        Z: Number of channels in the input.
        H: Number of heads in the multi-head attention.
        desc_q: Descriptor for the query tensor.
        desc_k: Descriptor for the key tensor.
        desc_v: Descriptor for the value tensor.
        desc_q_t: Descriptor for the transposed query tensor.
        desc_k_t: Descriptor for the transposed key tensor.
        desc_v_t: Descriptor for the transposed value tensor.
        desc_o: Descriptor for the output tensor.
        desc_o_t: Descriptor for the transposed output tensor.
        Mask: Attention mask tensor.
        dropout_p: Dropout probability.
        philox_seed: Seed for the Philox random number generator.
        stride_mz: Stride for the mask in the z dimension.
        stride_mh: Stride for the mask in the h dimension.
        stride_mm: Stride for the mask in the m dimension.
        stride_mn: Stride for the mask in the n dimension.
        N_CTX: Context length.
        HEAD_DIM: Dimension of each head.
        BLOCK_M: Block size for the queries.
        BLOCK_N: Block size for the keys/values.
        FP8_OUTPUT: Flag indicating if FP8 output is used.
        STAGE: Stage of the computation.
        warp_specialize: Flag indicating if warp specialization is used.
        ENABLE_JVP: Flag indicating if JVP (Jacobian-vector product) is enabled.
        ENABLE_DROPOUT: Flag indicating if dropout is enabled.
        MASK_TYPE: Type of mask used (0: no mask, 1: boolean, 2: additive).
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)  # N = KV

    # Prepare metadata and indices
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float32  # For dot products
    start_m = tl.program_id(0)  # Which block (in the input query sequence) to process
    off_hz = tl.program_id(
        1
    )  # Which head and batch element to process, with a program being a single head of a single batch element
    off_z = (
        off_hz // H
    )  # Which batch element this program is assigned to (n.b., each batch element has H heads)
    off_h = off_hz % H  # The position of the head to process in the batch

    # Initialize tensor descriptors
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],  # M = Q
    )
    if FP8_OUTPUT:
        v_shape = [HEAD_DIM, y_dim]
        v_strides = [N_CTX, 1]
        v_block_shape = [HEAD_DIM, BLOCK_N]
    else:
        v_shape = [y_dim, HEAD_DIM]
        v_strides = [HEAD_DIM, 1]
        v_block_shape = [BLOCK_N, HEAD_DIM]
    desc_v = _maybe_make_tensor_desc(
        desc_v, shape=v_shape, strides=v_strides, block_shape=v_block_shape
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # Initialize block pointer for the mask, if provided
    if MASK_TYPE > 0:
        mask_offset = off_z * stride_mz + off_h * stride_mh
        mask_block_ptr = tl.make_block_ptr(
            base=Mask + mask_offset,
            shape=(N_CTX, N_CTX),
            strides=(stride_mm, stride_mn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        mask_block_ptr = None

    # Initialize dropout offset for this block
    philox_offset_base = off_hz * N_CTX * N_CTX

    # Initialize offsets for the query tokens to process
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulator pointers:
    # m, the running maximum (one for each query)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l, the running sum (one for each query as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc, the output accumulator (one vector for each query)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if ENABLE_JVP:
        desc_q_t = _maybe_make_tensor_desc(
            desc_q_t,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        if FP8_OUTPUT:
            t_v_shape = [HEAD_DIM, y_dim]
            t_v_strides = [N_CTX, 1]
            t_v_block_shape = [HEAD_DIM, BLOCK_N]
        else:
            t_v_shape = [y_dim, HEAD_DIM]
            t_v_strides = [HEAD_DIM, 1]
            t_v_block_shape = [BLOCK_N, HEAD_DIM]
        desc_v_t = _maybe_make_tensor_desc(
            desc_v_t, shape=t_v_shape, strides=t_v_strides, block_shape=t_v_block_shape
        )
        desc_k_t = _maybe_make_tensor_desc(
            desc_k_t,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        desc_o_t = _maybe_make_tensor_desc(
            desc_o_t,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        # Load t_q: It will stay in SRAM throughout.
        t_q = desc_q_t.load([qo_offset_y, 0])
        g_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        mu_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        p_tv_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    else:
        t_q = None
        desc_k_t = None
        desc_v_t = None
        # Allocate minimal dummy tensors to keep consistent the return signature of _attn_fwd_inner_tma
        g_acc = tl.zeros([1, 1], dtype=tl.float32)
        mu_i = tl.zeros([1], dtype=tl.float32)
        p_tv_acc = tl.zeros([1, 1], dtype=tl.float32)

    # Prepare scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # Load q: It will stay in SRAM throughout.
    q = desc_q.load([qo_offset_y, 0])

    # Stage: 3 if causal, else 1
    if STAGE == 1 or STAGE == 3:
        # NOTE: This step runs for non-causal attention or for the
        # blocks to the left of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner_tma(
            acc,
            g_acc,  #
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            desc_k,
            desc_v,  #
            desc_k_t,
            desc_v_t,  #
            offset_y,
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    if STAGE == 3:
        # NOTE: This step runs for the blocks to the
        # right of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner_tma(
            acc,
            g_acc,  #
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            desc_k,
            desc_v,  #
            desc_k_t,
            desc_v_t,  #
            offset_y,
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    # Epilogue
    empty_mask = l_i == 0.0
    # Unconditionally apply the mask to avoid data-dependent branching
    l_i = tl.where(empty_mask, 1.0, l_i)

    m_i = m_i + tl.where(
        # NOTE: This is needed to compute the logsumexp for the backward pass.
        empty_mask,
        0.0,
        tl.math.log2(l_i),
    )

    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(desc_o.dtype))

    if ENABLE_JVP:
        t_p_v = g_acc / l_i[:, None] - (mu_i / l_i)[:, None] * acc
        t_y_out = t_p_v + p_tv_acc / l_i[:, None]
        desc_o_t.store([qo_offset_y, 0], t_y_out.to(desc_o_t.dtype))


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  # noqa: E741
):
    """Preprocess output deltas for the backward attention pass.

    Args:
        O: Output tensor.
        DO: Gradient of the output tensor.
        Delta: Accumulated gradients.
        N_CTX: Context length.
        BLOCK_M: Block size for M dimension.
        HEAD_DIM: Head dimension size.
    """
    # Collect sequence, batch, and head indices
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)

    # Load outputs and gradients
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(
        tl.float32
    )
    delta = tl.sum(o * do, axis=1)

    # Write-back the intermediate delta results
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    DO,  #
    O,  #
    M,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    start_n,
    start_m,
    num_steps,  #
    CAUSAL_MASKING: tl.constexpr,
    mask_ptr,
    mask_stride_tok1,
    mask_stride_tok2,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_h = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_h[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d
    o_ptrs = O + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d

    if MASK_TYPE > 0:
        mask_ptr = (
            mask_ptr + offs_m[None, :] * mask_stride_tok1 + offs_n[:, None] * mask_stride_tok2
        )

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    dtype = tl.float32

    for _ in range(num_steps):
        qT = tl.load(qT_ptrs)

        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)

        pT = tl.math.exp2(qkT - m[None, :])

        if MASK_TYPE > 0:
            mask = tl.load(mask_ptr)
            if MASK_TYPE == 1:
                pT = tl.where(mask == 1, pT, 0.0)
            elif MASK_TYPE == 2:
                attend = mask != MASK_CONST
                pT = tl.where(attend, pT, 0.0)

        elif CAUSAL_MASKING:
            causal_mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(causal_mask, pT, 0.0)

        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + curr_m * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M1, BLOCK_N1, N_CTX
            )
            pT = pT * dropout_mask.to(pT.dtype) * dropout_scale

        ppT = pT
        ppT = ppT.to(dtype)
        do = tl.load(do_ptrs)
        dv += tl.dot(ppT, do.to(dtype)).to(do.dtype)

        # Fused delta: compute rowsum(o * do) inline instead of loading from D
        o_block = tl.load(o_ptrs)
        Di = tl.sum(o_block * do.to(tl.float32), axis=1)

        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)

        if ENABLE_DROPOUT:
            dpT = dpT * dropout_mask.to(dpT.dtype) * dropout_scale

        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(dtype)
        dk += tl.dot(dsT, tl.trans(qT).to(dtype)).to(qT.dtype)

        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        o_ptrs += step_m * stride_tok

        if MASK_TYPE > 0:
            mask_ptr += step_m * mask_stride_tok1

    return dk, dv


# The main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    Di,
    stride_tok,
    stride_d,  #
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    num_steps,  #
    CAUSAL_MASKING: tl.constexpr,
    mask_ptr,
    mask_stride_tok1,
    mask_stride_tok2,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_h = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_h[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_h[:, None] * stride_d

    if MASK_TYPE > 0:
        mask_ptr = (
            mask_ptr + offs_m[:, None] * mask_stride_tok1 + offs_n[None, :] * mask_stride_tok2
        )

    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    dtype = tl.float32

    for _ in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)

        p = tl.math.exp2(qk - m)

        if MASK_TYPE > 0:
            mask = tl.load(mask_ptr)
            if MASK_TYPE == 1:
                p = tl.where(mask == 1, p, 0.0)
            elif MASK_TYPE == 2:
                attend = mask != MASK_CONST
                p = tl.where(attend, p, 0.0)

        elif CAUSAL_MASKING:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(causal_mask, p, 0.0)

        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX + curr_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M2, BLOCK_N2, N_CTX
            )
            p = p * dropout_mask.to(p.dtype) * dropout_scale

        dp = tl.dot(do, vT).to(tl.float32)

        if ENABLE_DROPOUT:
            dp = dp * dropout_mask.to(dp.dtype) * dropout_scale

        ds = p * (dp - Di[:, None])

        ds = ds.to(dtype)
        dq += tl.dot(ds, tl.trans(kT).to(dtype)).to(kT.dtype)

        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok

        if MASK_TYPE > 0:
            mask_ptr += step_n * mask_stride_tok2

    return dq


@triton.jit
def _attn_bwd_causal(
    Q,
    K,
    V,
    sm_scale,  #
    O,
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    mask_stride_z,
    mask_stride_h,
    mask_stride_tok1,
    mask_stride_tok2,
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    mask_ptr,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    ENABLE_DROPOUT: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996

    start_block_id = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    delta_shared_offset = (off_hz * N_CTX).to(tl.int64)
    qkv_shared_offset = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h

    Q += qkv_shared_offset
    K += qkv_shared_offset
    V += qkv_shared_offset
    O += qkv_shared_offset
    DO += qkv_shared_offset
    DQ += qkv_shared_offset
    DK += qkv_shared_offset
    DV += qkv_shared_offset

    M += delta_shared_offset

    if MASK_TYPE > 0:
        mask_offset = off_z.to(tl.int64) * mask_stride_z + off_h.to(tl.int64) * mask_stride_h
        mask_ptr += mask_offset

    philox_offset_base = off_hz * N_CTX * N_CTX

    # ====== COMPUTE dK and dV ======
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

    start_n = start_block_id * BLOCK_N1
    start_m = start_n

    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_h = tl.arange(0, HEAD_DIM)

    k = tl.load(K + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk, dv,
        Q, k, v, DO, O, M,
        stride_tok, stride_d, N_CTX,
        MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        CAUSAL_MASKING=True,
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk, dv,
        Q, k, v, DO, O, M,
        stride_tok, stride_d, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        CAUSAL_MASKING=False,
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # ====== COMPUTE dQ ======
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR

    start_m = start_block_id * BLOCK_M2
    end_n = start_m + BLOCK_M2

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)
    o_block = tl.load(O + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)

    m = tl.load(M + offs_m)

    # Fused delta: compute inline instead of separate preprocess kernel
    Di = tl.sum(o_block * do.to(tl.float32), axis=1)

    m = m[:, None]

    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        dq, q, K, V, do, m, Di,
        stride_tok, stride_d, N_CTX,
        BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
        start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
        CAUSAL_MASKING=True,
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    end_n -= num_steps * MASK_BLOCK_N2

    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(
        dq, q, K, V, do, m, Di,
        stride_tok, stride_d, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, end_n - num_steps * BLOCK_N2, num_steps,
        CAUSAL_MASKING=False,
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    dq *= LN2
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dq_ptrs, dq)


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    O,
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    mask_stride_z,
    mask_stride_h,
    mask_stride_tok1,
    mask_stride_tok2,
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    mask_ptr,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    ENABLE_DROPOUT: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996

    start_block_id = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    delta_shared_offset = (off_hz * N_CTX).to(tl.int64)
    qkv_shared_offset = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h

    Q += qkv_shared_offset
    K += qkv_shared_offset
    V += qkv_shared_offset
    O += qkv_shared_offset
    DO += qkv_shared_offset
    DQ += qkv_shared_offset
    DK += qkv_shared_offset
    DV += qkv_shared_offset

    M += delta_shared_offset

    if MASK_TYPE > 0:
        mask_offset = off_z.to(tl.int64) * mask_stride_z + off_h.to(tl.int64) * mask_stride_h
        mask_ptr += mask_offset

    philox_offset_base = off_hz * N_CTX * N_CTX

    # ====== COMPUTE dK and dV ======
    start_n = start_block_id * BLOCK_N1

    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_h = tl.arange(0, HEAD_DIM)

    k = tl.load(K + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)

    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    start_m = 0
    num_steps = N_CTX // BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk, dv,
        Q, k, v, DO, O, M,
        stride_tok, stride_d, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        CAUSAL_MASKING=False,
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # ====== COMPUTE dQ ======
    start_m = start_block_id * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)
    o_block = tl.load(O + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)

    m = tl.load(M + offs_m)

    # Fused delta: compute inline instead of separate preprocess kernel
    Di = tl.sum(o_block * do.to(tl.float32), axis=1)

    m = m[:, None]

    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    dq = _attn_bwd_dq(
        dq, q, K, V, do, m, Di,
        stride_tok, stride_d, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps,
        CAUSAL_MASKING=False,
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    dq *= LN2
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dq_ptrs, dq)


class JVPAttn(Function):
    """JVP (Jacobian-Vector Product) for Attention Mechanism."""

    class Grid(NamedTuple):
        """Grid configuration for JVP Attention."""

        M_BLOCKS: int
        Z_H: int
        ONE: Literal[1]

    class FnCtx(FunctionCtx):
        """Function context for JVP Attention."""

        sm_scale: float
        HEAD_DIM_K: int
        causal: bool
        grid: JVPAttn.Grid
        mask_tensor: Tensor
        MASK_TYPE: int
        dropout_p: float
        philox_seed: int
        ENABLE_DROPOUT: bool

    class FwdOutCtxContrib(NamedTuple):
        """Forward output context contributions for JVP Attention."""

        o_t: Tensor | None
        M: Tensor
        grid: JVPAttn.Grid
        HEAD_DIM_K: int
        sm_scale: float
        mask_tensor: Tensor
        MASK_TYPE: int
        dropout_p: float
        philox_seed: int
        ENABLE_DROPOUT: bool

    class FwdOut(NamedTuple):
        """Forward output for JVP Attention."""

        o: Tensor
        ctx: JVPAttn.FwdOutCtxContrib

    class JVPOut(NamedTuple):
        """JVP output for JVP Attention."""

        o: Tensor
        ctx: None

    class BwdOut(NamedTuple):
        """Backward output for JVP Attention."""

        q: Tensor
        k: Tensor
        v: Tensor
        q_t: None
        k_t: None
        v_t: None
        attn_mask: None
        dropout_p: None
        causal: None
        sm_scale: None
        warp_specialize: None
        USE_TMA: None
        verify_attn_mask: None

    class Strides(NamedTuple):
        """Strides for JVP Attention."""

        z: int
        h: int
        n_ctx: int
        head_dim: int

    @staticmethod
    def forward(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_t: Tensor | None,
        k_t: Tensor | None,
        v_t: Tensor | None,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        sm_scale: float | None = None,
        warp_specialize: bool = True,
        USE_TMA: bool = True,
        verify_attn_mask: bool = True,
    ) -> JVPAttn.FwdOut:
        """Forward pass for JVP Attention.

        NOTE: The following warning(s) will be raised if `verify_attn_mask=True`
        and an attention mask with any all-null head is provided:
            `RuntimeWarning: overflow encountered in exp2.`

        Args:
            q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
            k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
            v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
            q_t: Optional tensor for query transpose.
            k_t: Optional tensor for key transpose.
            v_t: Optional tensor for value transpose.
            attn_mask: Optional attention mask of shape (Z, H, N_CTX, N_CTX).
                Two types of masks are supported. A boolean mask where a value
                of True indicates that the element should take part in attention,
                or a float mask of the same type as query, key, value that is added
                to the attention score. The constant `MASK_CONST` is used to
                indicate masked positions in the float mask. All other values
                denote unmasked positions.
            dropout_p: Dropout probability.
            causal: Whether the attention is causal.
            sm_scale: Optional scaling factor for softmax.
            warp_specialize: Whether to use warp specialization.
            USE_TMA: Whether to use TMA.
            verify_attn_mask: Whether to verify the correctness of the provided attention mask.

        Returns:
            Outputs of JVP Attention.
        """
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not currently supported in JVP attention.")

        # Collect metadata
        Z, H, N_CTX, HEAD_DIM_Q = q.shape
        HEAD_DIM_K = k.shape[-1]
        HEAD_DIM_V = v.shape[-1]  # NOTE: When v is in float8_e5m2 it is transposed.

        STAGE = 3 if causal else 1
        ENABLE_JVP = q_t is not None

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V, (
            "JVP attention requires HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V"
            f" but got HEAD_DIM_Q={HEAD_DIM_Q}, HEAD_DIM_K={HEAD_DIM_K}, HEAD_DIM_V={HEAD_DIM_V}"
        )
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}, (
            "JVP attention only supports HEAD_DIM_K in {16, 32, 64, 128, 256},"
            f" but got HEAD_DIM_K={HEAD_DIM_K}",
        )

        if causal and attn_mask is not None:
            raise ValueError("Causal attention does not support an attention mask.")
        if attn_mask is not None:
            assert attn_mask.shape == (
                Z,
                H,
                N_CTX,
                N_CTX,
            ), "The provided attention mask must have 4 dimensions (Z, H, N_CTX, N_CTX)."
            assert attn_mask.dtype in {
                torch.bool,
                q.dtype,
            }, "The attention mask must be of the dtype bool or that of the query tensor."

        # Initialize arguments and tensors
        if sm_scale is None:
            sm_scale = HEAD_DIM_K**-0.5

        o = torch.empty_like(q)
        o_t: Tensor | None = torch.empty_like(q_t) if ENABLE_JVP else None
        M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

        # Tune kernel for custom (e.g., AMD) targets
        extra_kern_args = {}

        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        if is_cuda() and warp_specialize:
            if HEAD_DIM_K >= 128 or ENABLE_JVP:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80

        if hasattr(triton, "set_allocator") and is_cuda():

            def alloc_fn(size: int, align: int, _):
                """Custom allocator function for Triton."""
                return torch.empty(size, dtype=torch.int8, device="cuda")

            triton.set_allocator(alloc_fn)

        def strides_zhnd(t: Tensor) -> JVPAttn.Strides:
            """Get strides for a tensor with shape (Z, H, N_CTX, HEAD_DIM)."""
            return JVPAttn.Strides(t.stride(0), t.stride(1), t.stride(2), t.stride(3))

        # Determine mask type
        if attn_mask is None:
            MASK_TYPE = 0
            mask_tensor = torch.empty(0, device=q.device, dtype=q.dtype)
            mask_strides = (0, 0, 0, 0)
        elif attn_mask.dtype == torch.bool:
            MASK_TYPE = 1
            mask_tensor = attn_mask.contiguous()
            mask_strides = strides_zhnd(mask_tensor)
            if verify_attn_mask:
                # Check if any head is all False
                assert mask_tensor.any(
                    dim=(-1, -2)
                ).all(), "The attention mask cannot be all False for any head."
        else:
            MASK_TYPE = 2
            mask_tensor = attn_mask.to(q.dtype).contiguous()
            mask_strides = strides_zhnd(mask_tensor)
            if verify_attn_mask:
                # Check if the mask contains -inf/inf/NaN or is all (or no) MASK_CONST for any head
                assert not torch.isinf(
                    mask_tensor
                ).any(), "The attention mask cannot contain -inf or inf."
                assert not torch.isnan(
                    mask_tensor
                ).any(), "The attention mask cannot contain NaNs."
                assert (
                    (mask_tensor != MASK_CONST).any(dim=(-1, -2)).all()
                ), f"The attention mask cannot be all {MASK_CONST} (the masking constant) for any head."

                if not (mask_tensor == MASK_CONST).any():
                    raise UserWarning(
                        f"The provided floating-point attention mask does not mask out any elements with {MASK_CONST} (the masking constant). Consider using this constant for correct masking behavior."
                    )

        # Prepare dropout arguments
        ENABLE_DROPOUT = dropout_p > 0.0
        if ENABLE_DROPOUT:
            philox_seed = torch.randint(0, 2**32, (1,), device=q.device, dtype=torch.int64).item()
        else:
            philox_seed = 0

        # Set up grid for kernel launch
        Z_H = Z * H

        def grid(META: dict[str, Any]) -> JVPAttn.Grid:
            """Determine grid configuration."""
            return JVPAttn.Grid(triton.cdiv(N_CTX, META["BLOCK_M"]), Z_H, 1)

        if USE_TMA and supports_tma():
            y_dim = Z_H * N_CTX

            # Determine block sizes before creating descriptors so block_shape matches
            if N_CTX >= 128 and HEAD_DIM_K >= 64:
                tma_BLOCK_M = min(128, N_CTX)
                tma_BLOCK_N = min(64, N_CTX)
                tma_num_stages = 4
                tma_num_warps = 8
            else:
                tma_BLOCK_M = MIN_SEQUENCE_LENGTH
                tma_BLOCK_N = MIN_SEQUENCE_LENGTH
                tma_num_stages = NUM_STAGES_OPTIONS[0]
                tma_num_warps = 4

            qo_block_shape = [tma_BLOCK_M, HEAD_DIM_K]
            kv_block_shape = [tma_BLOCK_N, HEAD_DIM_K]

            desc_q = TensorDescriptor(
                q,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=qo_block_shape,
            )
            desc_q_t = (
                desc_q
                if q_t is None
                else TensorDescriptor(
                    q_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=qo_block_shape,
                )
            )

            if q.dtype == torch.float8_e5m2:
                v_shape = [HEAD_DIM_K, y_dim]
                v_strides = [N_CTX, 1]
            else:
                v_shape = [y_dim, HEAD_DIM_K]
                v_strides = [HEAD_DIM_K, 1]
            desc_v = TensorDescriptor(
                v, shape=v_shape, strides=v_strides, block_shape=kv_block_shape
            )
            if q_t is not None and q_t.dtype == torch.float8_e5m2:
                t_v_shape = [HEAD_DIM_K, y_dim]
                t_v_strides = [q_t.shape[2], 1]
            else:
                t_v_shape = [y_dim, HEAD_DIM_K]
                t_v_strides = [HEAD_DIM_K, 1]
            desc_v_t = (
                desc_v
                if v_t is None
                else TensorDescriptor(
                    v_t, shape=t_v_shape, strides=t_v_strides, block_shape=kv_block_shape
                )
            )

            desc_k = TensorDescriptor(
                k,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=kv_block_shape,
            )
            desc_k_t = (
                desc_k
                if k_t is None
                else TensorDescriptor(
                    k_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=kv_block_shape,
                )
            )

            desc_o = TensorDescriptor(
                o,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=qo_block_shape,
            )
            desc_o_t = (
                desc_o
                if o_t is None
                else TensorDescriptor(
                    o_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=qo_block_shape,
                )
            )

            _attn_fwd_tma[grid](
                sm_scale,
                M,  #
                Z,
                H,  #
                desc_q,
                desc_k,
                desc_v,  #
                desc_q_t,
                desc_k_t,
                desc_v_t,  #
                desc_o,
                desc_o_t,  #
                mask_tensor,  #
                dropout_p,  #
                philox_seed,  #
                *mask_strides,  #
                N_CTX=N_CTX,  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=STAGE,  #
                warp_specialize=warp_specialize,  #
                ENABLE_JVP=ENABLE_JVP,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                MASK_TYPE=MASK_TYPE,
                BLOCK_M=tma_BLOCK_M,  #
                BLOCK_N=tma_BLOCK_N,  #
                num_stages=tma_num_stages,  #
                num_warps=tma_num_warps,  #
                **extra_kern_args,
            )

        else:
            _attn_fwd[grid](
                q,
                k,
                v,
                q_t,
                k_t,
                v_t,  #
                sm_scale,
                M,
                o,
                o_t,  #
                mask_tensor,  #
                dropout_p,  #
                philox_seed,  #
                *strides_zhnd(q),  #
                *strides_zhnd(k),  #
                *strides_zhnd(v),  #
                *strides_zhnd(q if q_t is None else q_t),  #
                *strides_zhnd(k if k_t is None else k_t),  #
                *strides_zhnd(v if v_t is None else v_t),  #
                *strides_zhnd(o),  #
                *strides_zhnd(o if o_t is None else o_t),  #
                *mask_strides,  #
                Z,
                H,  #
                N_CTX=N_CTX,  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=STAGE,  #
                warp_specialize=warp_specialize,  #
                ENABLE_JVP=ENABLE_JVP,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                MASK_TYPE=MASK_TYPE,
                # NOTE: The following are safe (unit-tested) default values
                BLOCK_M=MIN_SEQUENCE_LENGTH,  #
                BLOCK_N=MIN_SEQUENCE_LENGTH,  #
                num_stages=NUM_STAGES_OPTIONS[0],  #
                num_warps=4,  #
                **extra_kern_args,
            )

        return JVPAttn.FwdOut(
            o,
            JVPAttn.FwdOutCtxContrib(
                o_t,
                M,
                grid,
                HEAD_DIM_K,
                sm_scale,
                mask_tensor,
                MASK_TYPE,
                dropout_p,
                philox_seed,
                ENABLE_DROPOUT,
            ),
        )

    @staticmethod
    def setup_context(ctx: JVPAttn.FnCtx, inputs, outputs: JVPAttn.FwdOut) -> Tensor:
        """Set up the context for JVP Attention.

        Args:
            ctx: The context to set up
            inputs: The input tensors
            outputs: The output tensors
        """
        (
            q,
            k,
            v,
            q_t,
            k_t,
            v_t,
            attn_mask,
            dropout_p,
            causal,
            sm_scale,
            warp_specialize,
            USE_TMA,
            verify_attn_mask,
        ) = inputs

        o, (
            o_t,
            M,
            grid,
            HEAD_DIM_K,
            sm_scale,
            mask_tensor,
            MASK_TYPE,
            dropout_p,
            philox_seed,
            ENABLE_DROPOUT,
        ) = outputs

        ctx.grid = grid
        ctx.save_for_forward(o_t)
        ctx.save_for_backward(q, k, v, o, M)

        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM_K = HEAD_DIM_K
        ctx.causal = causal
        ctx.mask_tensor = mask_tensor
        ctx.MASK_TYPE = MASK_TYPE
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.ENABLE_DROPOUT = ENABLE_DROPOUT

    @staticmethod
    def fwd(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        sm_scale: float | None = None,
        warp_specialize: bool = True,
        USE_TMA: bool = True,
    ) -> Tensor:
        """Forward pass for JVP Attention.

        NOTE: This is not an autograd convention. It's a workaround to get type-hinting and kwarg support.

        NOTE: Calls to `contiguous()` are necessary to ensure the inputs are contiguous in memory
        (e.g., due to an `unbind` call to create `q`, `k`, `v`) but nonetheless may incur a performance cost.

        Args:
            q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
            k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
            v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
            attn_mask: Optional attention mask of shape (Z, H, N_CTX, N_CTX). Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score.
            dropout_p: Dropout probability.
            causal: Whether to use causal attention.
            sm_scale: The softmax scale factor.
            warp_specialize: Whether to use warp specialization.
            USE_TMA: Whether to use TMA.

        Returns:
            The output tensor.
        """
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        out: JVPAttn.FwdOut = JVPAttn.apply(
            q,
            k,
            v,
            None,
            None,
            None,
            attn_mask,
            dropout_p,
            causal,
            sm_scale,
            warp_specialize,
            USE_TMA,
        )

        a, _ = out
        return a

    @staticmethod
    def fwd_dual(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        sm_scale: float | None = None,
        warp_specialize: bool = True,
        USE_TMA: bool = True,
    ) -> Tensor:
        """Forward pass for JVP Attention with dual tensor inputs.

        NOTE: This is not an autograd convention. It's a workaround to get type-hinting and kwarg support.

        NOTE: Calls to `contiguous()` are necessary to ensure the inputs are contiguous in memory
        (e.g., due to an `unbind` call to create `q`, `k`, `v`) but nonetheless may incur a performance cost.

        Args:
            q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
            k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
            v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
            attn_mask: Optional attention mask of shape (Z, H, N_CTX, N_CTX). Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score.
            dropout_p: Dropout probability.
            causal: Whether to use causal attention.
            sm_scale: The softmax scale factor.
            warp_specialize: Whether to use warp specialization.
            USE_TMA: Whether to use TMA.

        Returns:
            The output tensor.
        """
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        q_p, q_t = fwAD.unpack_dual(q)
        k_p, k_t = fwAD.unpack_dual(k)
        v_p, v_t = fwAD.unpack_dual(v)

        # NOTE: We pass some dualtensor args to ensure jvp() will be called,
        # but we also pass tangents separately, as forward() demotes dual
        # tensor args to primals for some reason.
        out: JVPAttn.FwdOut = JVPAttn.apply(
            q,
            k,
            v,
            q_t,
            k_t,
            v_t,
            attn_mask,
            dropout_p,
            causal,
            sm_scale,
            warp_specialize,
            USE_TMA,
        )

        a, _ = out
        return a

    @staticmethod
    def jvp(ctx: JVPAttn.FnCtx, gq: Tensor, gk: Tensor, gv: Tensor, *_) -> JVPAttn.JVPOut:
        """Compute the Jacobian-vector product (JVP) for JVP Attention.

        Args:
            ctx: The context
            gq: The gradient of the query tensor
            gk: The gradient of the key tensor
            gv: The gradient of the value tensor

        Returns:
            The JVP output.
        """
        return JVPAttn.JVPOut(ctx.saved_for_forward[0], None)

    @staticmethod
    def backward(ctx, do, _) -> JVPAttn.BwdOut:
        q, k, v, o, M = ctx.saved_tensors

        return JVPAttnBwd.apply(
            do, q, k, v, o, M, ctx.mask_tensor, 
            ctx.HEAD_DIM_K, ctx.MASK_TYPE, ctx.causal, ctx.dropout_p,
            ctx.ENABLE_DROPOUT, ctx.sm_scale, ctx.philox_seed
        )
    
split_configs = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8, maxnreg=168),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8, maxnreg=168),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8, maxnreg=232),
]


@triton.jit
def _double_bwd_q_inner(
    q, gq, do, lse,
    K_ptr, V_ptr, GK_ptr, GV_ptr, mask_ptr,
    stride_kn, stride_kk, stride_vn, stride_vk,
    mask_stride_tok1, mask_stride_tok2,
    sm_scale_2, sm_scale_e, drop_scale, dropout_p, philox_seed,
    philox_offset_base, start_m, offs_m, offs_d,
    acc_ssm, acc_beta, acc_tmp1, acc_tmp2,
    acc_dq, acc_dgout,
    N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
    MASK_TYPE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PASS_ID: tl.constexpr,
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    offs_n_init = tl.arange(0, BLOCK_N)
    base_kn = offs_n_init[:, None] * stride_kn + offs_d[None, :] * stride_kk
    base_vn = offs_n_init[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k_ptrs = K_ptr + base_kn
    v_ptrs = V_ptr + base_vn
    gk_ptrs = GK_ptr + base_kn
    gv_ptrs = GV_ptr + base_vn
    step_n_k = BLOCK_N * stride_kn
    step_n_v = BLOCK_N * stride_vn

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        gk = tl.load(gk_ptrs)
        gv = tl.load(gv_ptrs)

        kT = tl.trans(k)
        qk = tl.dot(q, kT) * sm_scale_2

        if MASK_TYPE > 0:
            mask_vals = tl.load(mask_ptr + offs_m[:, None] * mask_stride_tok1 + offs_n[None, :] * mask_stride_tok2)
            if MASK_TYPE == 1:
                qk = tl.where(mask_vals == 1, qk, float("-inf"))
            elif MASK_TYPE == 2:
                qk = tl.where(mask_vals != MASK_CONST, qk, float("-inf"))

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        p = tl.math.exp2(qk - lse[:, None])

        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + (start_m * BLOCK_M) * N_CTX + start_n
            drop_mask, _ = create_dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX)
        else:
            drop_mask = None

        vT = tl.trans(v)
        dp_raw = tl.dot(do, vT)
        dp_drop = dp_raw * drop_mask * drop_scale if ENABLE_DROPOUT else dp_raw

        b_ds = (tl.dot(gq, kT) + tl.dot(q, tl.trans(gk))) * sm_scale_e

        if PASS_ID == 0:
            v_dp_raw = tl.dot(do, tl.trans(gv))
            v_dp_drop = v_dp_raw * drop_mask * drop_scale if ENABLE_DROPOUT else v_dp_raw
            acc_ssm += tl.sum(dp_drop * p, axis=1)
            acc_beta += tl.sum(b_ds * p, axis=1)
            acc_tmp1 += tl.sum(v_dp_drop * p, axis=1)
            acc_tmp2 += tl.sum(dp_drop * p * b_ds, axis=1)
        else:
            p_drop = p * drop_mask * drop_scale if ENABLE_DROPOUT else p
            v_p = p * (b_ds - acc_beta[:, None])
            v_p_drop = v_p * drop_mask * drop_scale if ENABLE_DROPOUT else v_p

            v_dp = tl.dot(do, tl.trans(gv))
            v_dp = v_dp * drop_mask * drop_scale if ENABLE_DROPOUT else v_dp
            tmp_val = acc_tmp1 + acc_tmp2 - acc_ssm * acc_beta
            v_ds = (v_dp - tmp_val[:, None]) * p + (dp_drop - acc_ssm[:, None]) * v_p
            ds = (dp_drop - acc_ssm[:, None]) * p

            acc_dgout += tl.dot(v_p_drop.to(v.dtype), v) + tl.dot(p_drop.to(gv.dtype), gv)
            acc_dq += (tl.dot(v_ds.to(k.dtype), k) + tl.dot(ds.to(gk.dtype), gk)) * sm_scale_e

        k_ptrs += step_n_k
        v_ptrs += step_n_v
        gk_ptrs += step_n_k
        gv_ptrs += step_n_v

    return acc_ssm, acc_beta, acc_tmp1, acc_tmp2, acc_dq, acc_dgout


@triton.autotune(configs=split_configs, key=['N_CTX', 'H'])
@triton.jit
def _attn_double_bwd_q_kernel(
    Q, K, V, DO, LSE,
    GQ, GK, GV,
    DGRAD_Q, DGRAD_GOUT,
    SSM_OUT, BETA_OUT, TMP_OUT,
    stride_z, stride_h, stride_m, stride_d,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    mask_ptr, mask_stride_z, mask_stride_h, mask_stride_tok1, mask_stride_tok2,
    sm_scale_2, sm_scale_e, drop_scale, dropout_p, philox_seed,
    H: tl.constexpr, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr, ENABLE_DROPOUT: tl.constexpr, MASK_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MASK_CONST: tl.constexpr = MASK_CONST
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    philox_offset_base = off_hz * N_CTX * N_CTX

    q_base = off_z * stride_z + off_h * stride_h
    k_base = off_z * stride_kz + off_h * stride_kh
    v_base = off_z * stride_vz + off_h * stride_vh

    Q_ptr = Q + q_base
    GQ_ptr = GQ + q_base
    DO_ptr = DO + q_base
    DGRAD_Q_ptr = DGRAD_Q + q_base
    DGRAD_GOUT_ptr = DGRAD_GOUT + q_base
    K_ptr = K + k_base
    GK_ptr = GK + k_base
    V_ptr = V + v_base
    GV_ptr = GV + v_base

    mask_ptr_local = mask_ptr
    if MASK_TYPE > 0:
        mask_ptr_local += off_z * mask_stride_z + off_h * mask_stride_h

    lse_base = off_hz * N_CTX
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q_ptr + (offs_m[:, None] * stride_m + offs_d[None, :] * stride_d))
    gq = tl.load(GQ_ptr + (offs_m[:, None] * stride_m + offs_d[None, :] * stride_d))
    do = tl.load(DO_ptr + (offs_m[:, None] * stride_m + offs_d[None, :] * stride_d))
    lse = tl.load(LSE + lse_base + offs_m)

    acc_ssm = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_beta = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_tmp1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_tmp2 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    acc_dgout = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Pass 1: accumulate scalars (ssm, beta, tmp)
    acc_ssm, acc_beta, acc_tmp1, acc_tmp2, acc_dq, acc_dgout = _double_bwd_q_inner(
        q, gq, do, lse,
        K_ptr, V_ptr, GK_ptr, GV_ptr, mask_ptr_local,
        stride_kn, stride_kk, stride_vn, stride_vk,
        mask_stride_tok1, mask_stride_tok2,
        sm_scale_2, sm_scale_e, drop_scale, dropout_p, philox_seed,
        philox_offset_base, start_m, offs_m, offs_d,
        acc_ssm, acc_beta, acc_tmp1, acc_tmp2,
        acc_dq, acc_dgout,
        N_CTX, HEAD_DIM,
        IS_CAUSAL, ENABLE_DROPOUT, MASK_TYPE, BLOCK_M, BLOCK_N,
        PASS_ID=0,
    )

    tl.store(SSM_OUT + lse_base + offs_m, acc_ssm)
    tl.store(BETA_OUT + lse_base + offs_m, acc_beta)
    tl.store(TMP_OUT + lse_base + offs_m, acc_tmp1 + acc_tmp2 - acc_ssm * acc_beta)

    # Pass 2: compute dQ, dGout using the scalars from pass 1 (still in registers)
    acc_ssm, acc_beta, acc_tmp1, acc_tmp2, acc_dq, acc_dgout = _double_bwd_q_inner(
        q, gq, do, lse,
        K_ptr, V_ptr, GK_ptr, GV_ptr, mask_ptr_local,
        stride_kn, stride_kk, stride_vn, stride_vk,
        mask_stride_tok1, mask_stride_tok2,
        sm_scale_2, sm_scale_e, drop_scale, dropout_p, philox_seed,
        philox_offset_base, start_m, offs_m, offs_d,
        acc_ssm, acc_beta, acc_tmp1, acc_tmp2,
        acc_dq, acc_dgout,
        N_CTX, HEAD_DIM,
        IS_CAUSAL, ENABLE_DROPOUT, MASK_TYPE, BLOCK_M, BLOCK_N,
        PASS_ID=1,
    )

    tl.store(DGRAD_Q_ptr + (offs_m[:, None] * stride_m + offs_d[None, :] * stride_d), acc_dq.to(q.dtype))
    tl.store(DGRAD_GOUT_ptr + (offs_m[:, None] * stride_m + offs_d[None, :] * stride_d), acc_dgout.to(q.dtype))


@triton.autotune(configs=split_configs, key=['N_CTX', 'H'])
@triton.jit
def _attn_double_bwd_kv_kernel(
    Q, K, V, DO, LSE,
    GQ, GK, GV,
    DGRAD_K, DGRAD_V,
    SSM_IN, BETA_IN, TMP_IN,
    stride_z, stride_h, stride_m, stride_d,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    mask_ptr, mask_stride_z, mask_stride_h, mask_stride_tok1, mask_stride_tok2,
    sm_scale_2, sm_scale_e, drop_scale, dropout_p, philox_seed,
    H: tl.constexpr, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr, ENABLE_DROPOUT: tl.constexpr, MASK_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MASK_CONST: tl.constexpr = MASK_CONST
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    philox_offset_base = off_hz * N_CTX * N_CTX

    q_base = off_z * stride_z + off_h * stride_h
    k_base = off_z * stride_kz + off_h * stride_kh
    v_base = off_z * stride_vz + off_h * stride_vh

    Q_ptr = Q + q_base
    GQ_ptr = GQ + q_base
    DO_ptr = DO + q_base
    K_ptr = K + k_base
    GK_ptr = GK + k_base
    DGRAD_K_ptr = DGRAD_K + k_base
    V_ptr = V + v_base
    GV_ptr = GV + v_base
    DGRAD_V_ptr = DGRAD_V + v_base

    mask_ptr_local = mask_ptr
    if MASK_TYPE > 0:
        mask_ptr_local += off_z * mask_stride_z + off_h * mask_stride_h

    lse_base = off_hz * N_CTX
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k = tl.load(K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk))
    v = tl.load(V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk))
    gk = tl.load(GK_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk))
    gv = tl.load(GV_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk))

    acc_dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    acc_dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    kT = tl.trans(k)
    vT = tl.trans(v)
    gkT = tl.trans(gk)
    gvT = tl.trans(gv)

    lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    base_qgd = offs_m_init[:, None] * stride_m + offs_d[None, :] * stride_d
    q_ptrs = Q_ptr + base_qgd
    gq_ptrs = GQ_ptr + base_qgd
    do_ptrs = DO_ptr + base_qgd
    lse_ptrs = LSE + lse_base + offs_m_init
    ssm_ptrs = SSM_IN + lse_base + offs_m_init
    beta_ptrs = BETA_IN + lse_base + offs_m_init
    tmp_ptrs = TMP_IN + lse_base + offs_m_init
    step_m = BLOCK_M * stride_m

    for start_m in range(lo, N_CTX, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(q_ptrs)
        gq = tl.load(gq_ptrs)
        do = tl.load(do_ptrs)

        lse = tl.load(lse_ptrs)
        ssm = tl.load(ssm_ptrs)
        beta = tl.load(beta_ptrs)
        tmp = tl.load(tmp_ptrs)

        qk = tl.dot(q, kT) * sm_scale_2

        if MASK_TYPE > 0:
            mask_vals = tl.load(mask_ptr_local + offs_m[:, None] * mask_stride_tok1 + offs_n[None, :] * mask_stride_tok2)
            if MASK_TYPE == 1:
                qk = tl.where(mask_vals == 1, qk, float("-inf"))
            elif MASK_TYPE == 2:
                qk = tl.where(mask_vals != MASK_CONST, qk, float("-inf"))

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        p = tl.math.exp2(qk - lse[:, None])

        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX + start_n * BLOCK_N
            drop_mask, _ = create_dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX)
            dp = tl.dot(do, vT) * drop_mask * drop_scale
        else:
            dp = tl.dot(do, vT)

        b_ds = (tl.dot(gq, kT) + tl.dot(q, gkT)) * sm_scale_e
        v_p = p * (b_ds - beta[:, None])
        v_p_drop = v_p * drop_mask * drop_scale if ENABLE_DROPOUT else v_p

        v_dp = tl.dot(do, gvT)
        v_dp = v_dp * drop_mask * drop_scale if ENABLE_DROPOUT else v_dp
        v_ds = (v_dp - tmp[:, None]) * p + (dp - ssm[:, None]) * v_p
        ds = (dp - ssm[:, None]) * p

        acc_dk += (tl.dot(tl.trans(v_ds).to(q.dtype), q) + tl.dot(tl.trans(ds).to(gq.dtype), gq)) * sm_scale_e
        acc_dv += tl.dot(tl.trans(v_p_drop).to(do.dtype), do)

        q_ptrs += step_m
        gq_ptrs += step_m
        do_ptrs += step_m
        lse_ptrs += BLOCK_M
        ssm_ptrs += BLOCK_M
        beta_ptrs += BLOCK_M
        tmp_ptrs += BLOCK_M

    tl.store(DGRAD_K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk), acc_dk.to(k.dtype))
    tl.store(DGRAD_V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk), acc_dv.to(v.dtype))

class JVPAttnBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, do, q, k, v, o, M, mask_tensor, HEAD_DIM_K, MASK_TYPE, causal, dropout_p, ENABLE_DROPOUT, sm_scale, philox_seed):
        # Save tensors needed for double-backward
        ctx.save_for_backward(do, q, k, v, M)
        
        # Save config
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM_K = HEAD_DIM_K
        ctx.causal = causal
        ctx.mask_tensor = mask_tensor
        ctx.MASK_TYPE = MASK_TYPE
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.ENABLE_DROPOUT = ENABLE_DROPOUT

        """Backward pass for JVP Attention.

        NOTE: A call to `contiguous()` may be necessary to ensure the output derivatives are contiguous
        in memory (e.g., due to autograd weirdness) but nonetheless may incur a performance cost.

        Args:
            ctx: The context
            do: The gradient of the output tensor

        Returns:
            The backward output.
        """

        # Ensure inputs/outputs the kernel reads share the same (contiguous) layout
        if not (
            q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and o.is_contiguous()
        ):
            raise ValueError(
                "JVPAttn expected q, k, v, o to be contiguous; got "
                f"q.is_contiguous()={q.is_contiguous()}, k.is_contiguous()={k.is_contiguous()}, "
                f"v.is_contiguous()={v.is_contiguous()}, o.is_contiguous()={o.is_contiguous()}, "
                f"do.is_contiguous()={do.is_contiguous()}"
            )

        # NOTE: Autograd may deliver a non-contiguous output gradient; if so, normalize it.
        if not do.is_contiguous():
            do = do.contiguous()

        # Ensure all inputs/outputs the kernel reads share the same layout
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride(), (
            "JVPAttn expected q, k, v, o, do to have the same layout; got "
            f"q.stride()={q.stride()}, k.stride()={k.stride()}, v.stride()={v.stride()}, "
            f"o.stride()={o.stride()}, do.stride()={do.stride()}"
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        Z, H, N_CTX = q.shape[:3]

        BLK_SLICE_FACTOR = 2
        BLOCK_MIN = MIN_SEQUENCE_LENGTH
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = BLOCK_MIN, BLOCK_MIN, BLOCK_MIN, BLOCK_MIN

        assert N_CTX % BLOCK_MIN == 0, f"N_CTX must be divisible by BLOCK_MIN={BLOCK_MIN}"

        if not ctx.causal:
            assert (
                BLOCK_M1 == BLOCK_M2 == BLOCK_N1 == BLOCK_N2
            ), "For non-causal attention, all block sizes must be equal."

        RCP_LN2 = 1.4426950408889634
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)

        if ctx.MASK_TYPE == 0:
            mask_strides = (0, 0, 0, 0)
        else:
            mask_strides = (
                ctx.mask_tensor.stride(0),
                ctx.mask_tensor.stride(1),
                ctx.mask_tensor.stride(2),
                ctx.mask_tensor.stride(3),
            )

        Z_H = Z * H

        # Delta precompute is fused into the backward kernel — no separate
        # _attn_bwd_preprocess launch needed. The backward kernel computes
        # delta = rowsum(o * do) inline for each block it processes.
        grid = (N_CTX // BLOCK_MIN, Z_H)
        bwd_kernel = _attn_bwd_causal if ctx.causal else _attn_bwd
        num_stages = (
            5
            if is_cuda() and torch.cuda.get_device_capability()[0] >= 9
            else NUM_STAGES_OPTIONS[0]
        )

        bwd_kernel[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            o,
            do,
            dq,
            dk,
            dv,  #
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            mask_strides[0],
            mask_strides[1],
            mask_strides[2],
            mask_strides[3],  #
            H,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM_K,  #
            mask_ptr=ctx.mask_tensor,
            MASK_TYPE=ctx.MASK_TYPE,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            ENABLE_DROPOUT=ctx.ENABLE_DROPOUT,
            num_stages=num_stages,  #
            num_warps=4,  #
        )

        return JVPAttn.BwdOut(
            dq, dk, dv, None, None, None, None, None, None, None, None, None, None
        )
    
    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v, *_):
        grad_out, q, k, v, M_LSE = ctx.saved_tensors
        Z, H, N_CTX, HEAD_DIM = q.shape

        # Enforce contiguous memory to prevent 0-stride out-of-bounds reads
        grad_out = grad_out.contiguous()
        grad_q = grad_q.contiguous() if grad_q is not None else torch.zeros_like(q)
        grad_k = grad_k.contiguous() if grad_k is not None else torch.zeros_like(k)
        grad_v = grad_v.contiguous() if grad_v is not None else torch.zeros_like(v)
        
        scale_e = ctx.sm_scale if ctx.sm_scale is not None else (HEAD_DIM ** -0.5)
        scale_2 = scale_e * 1.44269504 # base-2 for hardware exp2
        drop_scale = 1.0 / (1.0 - ctx.dropout_p) if ctx.dropout_p < 1.0 else 0.0
        
        dgrad_q = torch.zeros_like(q)
        dgrad_grad_out = torch.zeros_like(grad_out)
        dgrad_k = torch.zeros_like(k)
        dgrad_v = torch.zeros_like(v)

        # Calculate mask strides
        if ctx.MASK_TYPE == 0:
            mask_strides = (0, 0, 0, 0)
        else:
            mask_strides = (
                ctx.mask_tensor.stride(0), ctx.mask_tensor.stride(1),
                ctx.mask_tensor.stride(2), ctx.mask_tensor.stride(3),
            )

        # Buffers zero-initialized
        ssm_buffer = torch.zeros((Z, H, N_CTX), device=q.device, dtype=torch.float32)
        beta_buffer = torch.zeros((Z, H, N_CTX), device=q.device, dtype=torch.float32)
        tmp_buffer = torch.zeros((Z, H, N_CTX), device=q.device, dtype=torch.float32)

        # 1. Launch Q-Kernel 
        grid_q = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), Z * H)
        _attn_double_bwd_q_kernel[grid_q](
            q, k, v, grad_out, M_LSE, 
            grad_q, grad_k, grad_v,
            dgrad_q, dgrad_grad_out,
            ssm_buffer, beta_buffer, tmp_buffer,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            ctx.mask_tensor, mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
            scale_2, scale_e, drop_scale, ctx.dropout_p, ctx.philox_seed,
            H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=ctx.causal, ENABLE_DROPOUT=ctx.ENABLE_DROPOUT, MASK_TYPE=ctx.MASK_TYPE
        )

        # 2. Launch KV-Kernel
        grid_kv = lambda META: (triton.cdiv(N_CTX, META['BLOCK_N']), Z * H)
        _attn_double_bwd_kv_kernel[grid_kv](
            q, k, v, grad_out, M_LSE, 
            grad_q, grad_k, grad_v,
            dgrad_k, dgrad_v,
            ssm_buffer, beta_buffer, tmp_buffer,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            ctx.mask_tensor, mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
            scale_2, scale_e, drop_scale, ctx.dropout_p, ctx.philox_seed,
            H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
            IS_CAUSAL=ctx.causal, ENABLE_DROPOUT=ctx.ENABLE_DROPOUT, MASK_TYPE=ctx.MASK_TYPE
        )

        return dgrad_grad_out, dgrad_q, dgrad_k, dgrad_v, None, None, None, None, None, None, None, None, None, None


def hvp_fwd_over_rev(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    v_q: torch.Tensor,
    v_k: torch.Tensor,
    v_v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Hessian-Vector Product (HVP) for Attention via Forward-over-Reverse AD.

    This eliminates the standard backward pass (with create_graph=True) by computing
    the forward pass to get the log-sum-exp (M), then directly launching the
    double-backward kernels with the tangent vectors. This saves ~2 passes compared
    to the reverse-over-reverse approach in hvp_semi_manual.

    Flow: forward (1 pass) → double-backward kernels (3 passes) = 4 passes total
    vs reverse-over-reverse: forward + backward(create_graph) + double-backward = 6 passes
    """
    sm_scale = scaling if scaling is not None else query.shape[-1] ** -0.5

    q = query.detach().contiguous()
    k = key.detach().contiguous()
    v = value.detach().contiguous()

    # Step 1: Forward pass to get O and M (log-sum-exp)
    with torch.no_grad():
        out = JVPAttn.apply(
            q, k, v, None, None, None, attention_mask, 0.0, is_causal,
            sm_scale, True, True, False,
        )
        _, ctx_contrib = out
        M_LSE = ctx_contrib.M
        mask_tensor = ctx_contrib.mask_tensor
        MASK_TYPE = ctx_contrib.MASK_TYPE

    # Step 2: Launch double-backward kernels directly (skip standard backward)
    Z, H, N_CTX, HEAD_DIM = q.shape

    grad_out = grad_out.contiguous()
    v_q = v_q.contiguous()
    v_k = v_k.contiguous()
    v_v = v_v.contiguous()

    scale_e = sm_scale
    scale_2 = scale_e * 1.44269504
    drop_scale = 1.0

    dgrad_q = torch.zeros_like(q)
    dgrad_grad_out = torch.zeros_like(grad_out)
    dgrad_k = torch.zeros_like(k)
    dgrad_v = torch.zeros_like(v)

    if MASK_TYPE == 0:
        mask_strides = (0, 0, 0, 0)
    else:
        mask_strides = (
            mask_tensor.stride(0), mask_tensor.stride(1),
            mask_tensor.stride(2), mask_tensor.stride(3),
        )

    ssm_buffer = torch.zeros((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    beta_buffer = torch.zeros((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    tmp_buffer = torch.zeros((Z, H, N_CTX), device=q.device, dtype=torch.float32)

    grid_q = lambda META: (triton.cdiv(N_CTX, META['BLOCK_M']), Z * H)
    _attn_double_bwd_q_kernel[grid_q](
        q, k, v, grad_out, M_LSE,
        v_q, v_k, v_v,
        dgrad_q, dgrad_grad_out,
        ssm_buffer, beta_buffer, tmp_buffer,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask_tensor, mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
        scale_2, scale_e, drop_scale, 0.0, 0,
        H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
        IS_CAUSAL=is_causal, ENABLE_DROPOUT=False, MASK_TYPE=MASK_TYPE,
    )

    grid_kv = lambda META: (triton.cdiv(N_CTX, META['BLOCK_N']), Z * H)
    _attn_double_bwd_kv_kernel[grid_kv](
        q, k, v, grad_out, M_LSE,
        v_q, v_k, v_v,
        dgrad_k, dgrad_v,
        ssm_buffer, beta_buffer, tmp_buffer,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask_tensor, mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
        scale_2, scale_e, drop_scale, 0.0, 0,
        H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
        IS_CAUSAL=is_causal, ENABLE_DROPOUT=False, MASK_TYPE=MASK_TYPE,
    )

    return dgrad_q, dgrad_k, dgrad_v


def hvp_semi_manual(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    v_q: torch.Tensor,
    v_k: torch.Tensor,
    v_v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Hessian-Vector Product (HVP) for Attention via Reverse-over-Reverse AD.
    """
    # 1. Ensure tensors require gradients to build the computational graph
    query = query.requires_grad_(True)
    key = key.requires_grad_(True)
    value = value.requires_grad_(True)

    # set dropout to 0 (not supported by JVP/HVP)
    dropout = 0.0
    
    # 2. Forward Pass (Triton)
    attn_output = JVPAttn.fwd(
        query, key, value, attention_mask, dropout, is_causal, scaling,
    )

    # 3. First Backward Pass: Vector-Jacobian Product (VJP)
    # This calls _attn_bwd (Triton) and creates a graph connecting the gradients
    # back to q, k, v using JVPAttnBwd.forward
    dq, dk, dv = torch.autograd.grad(
        outputs=attn_output,
        inputs=(query, key, value),
        grad_outputs=grad_out,
        create_graph=True, # Critical: enables the second backward pass
        retain_graph=True
    )

    # 4. Second Backward Pass: Hessian-Vector Product (HVP)
    # This triggers JVPAttnBwd.backward (pure PyTorch math) to compute the derivatives
    # of the gradients in the direction of the tangents (v_q, v_k, v_v)
    hvp_q, hvp_k, hvp_v = torch.autograd.grad(
        outputs=(dq, dk, dv),
        inputs=(query, key, value),
        grad_outputs=(v_q, v_k, v_v),
        retain_graph=True
    )

    return hvp_q, hvp_k, hvp_v