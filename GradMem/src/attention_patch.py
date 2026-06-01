from __future__ import annotations
import warnings
from contextlib import contextmanager
from typing import Optional
import torch
import torch.nn.functional as F
from hvp_triton import JVPAttn, hvp_fwd_over_rev
_HVP_AVAILABLE = True

# Constraints imposed by the JVPAttn kernel (from hvp_triton.py assertions).
_SUPPORTED_HEAD_DIMS = {16, 32, 64, 128, 256}
_SUPPORTED_DTYPES    = {torch.float16, torch.bfloat16}
_SEQ_LEN_MULTIPLE    = 32   # MIN_SEQUENCE_LENGTH in hvp_triton.py

_original_sdpa = None
_original_linear = None
_bypass_patch = False


def _patched_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Drop-in replacement for F.scaled_dot_product_attention.

    Routes to JVPAttn.fwd() when the call is compatible with the Triton
    kernel.  Falls back to the original SDPA otherwise.
    """
    if not _HVP_AVAILABLE or _bypass_patch:
        return _original_sdpa(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, **kwargs,
        )

    B, H, S_q, head_dim = query.shape
    S_k = key.shape[2]

    compatible = (
        query.dim() == 4
        and head_dim in _SUPPORTED_HEAD_DIMS
        and S_q % _SEQ_LEN_MULTIPLE == 0
        and S_k % _SEQ_LEN_MULTIPLE == 0
        and query.dtype in _SUPPORTED_DTYPES | {torch.float32}
        and dropout_p == 0.0
        and not (is_causal and attn_mask is not None)
    )

    if not compatible:
        return _original_sdpa(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, **kwargs,
        )

    jvp_mask = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            jvp_mask = attn_mask.expand(B, H, S_q, S_k).contiguous()
        else:
            from hvp_triton import MASK_CONST as _MASK_CONST
            jvp_mask = attn_mask.float().expand(B, H, S_q, S_k).contiguous()
            jvp_mask[(jvp_mask < -1e3)] = _MASK_CONST
            jvp_mask = jvp_mask.bfloat16()

    if query.dtype == torch.float32:
        query = query.bfloat16()
        key   = key.bfloat16()
        value = value.bfloat16()
        if jvp_mask is not None and not jvp_mask.dtype == torch.bool:
            jvp_mask = jvp_mask.bfloat16()

    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        query = query.contiguous()
        key   = key.contiguous()
        value = value.contiguous()

    return JVPAttn.fwd(
        query, key, value,
        attn_mask=jvp_mask,
        dropout_p=0.0,
        causal=is_causal,
        sm_scale=scale,
        warp_specialize=False,
    )


def patch_attention() -> bool:
    """
    Replace F.scaled_dot_product_attention with the JVPAttn-backed version.

    Call this once before instantiating or loading the model.  All subsequent
    nn.MultiheadAttention / HuggingFace transformer attention modules will use
    the patched SDPA, giving them efficient double-backward support.
    """
    global _original_sdpa

    if _original_sdpa is not None:
        # Already patched — idempotent.
        return _HVP_AVAILABLE

    if not _HVP_AVAILABLE:
        warnings.warn(
            f"HVP Triton kernel unavailable ({_HVP_IMPORT_ERROR}).\n"
            "GradMem will use PyTorch eager double-backward, which is correct "
            "but significantly slower and more memory-intensive for long contexts.\n"
            "See paper Appendix C for the performance difference.",
            RuntimeWarning,
            stacklevel=2,
        )

    _original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _patched_sdpa
    try:
        import torch.nn.functional as _F_alias
        _F_alias.scaled_dot_product_attention = _patched_sdpa
    except Exception:
        pass

    return _HVP_AVAILABLE


def unpatch_attention() -> None:
    """
    Restore F.scaled_dot_product_attention to the original PyTorch version.

    Safe to call even if patch_attention() was never called.
    """
    global _original_sdpa
    if _original_sdpa is not None:
        F.scaled_dot_product_attention = _original_sdpa
        try:
            import torch.nn.functional as _F_alias
            _F_alias.scaled_dot_product_attention = _original_sdpa
        except Exception:
            pass
        _original_sdpa = None


def is_patched() -> bool:
    """Return True if the attention patch is currently active."""
    return _original_sdpa is not None


@contextmanager
def bypass_kernel():
    """Temporarily route all SDPA calls to the original PyTorch implementation.
    """
    global _bypass_patch
    _bypass_patch = True
    try:
        yield
    finally:
        _bypass_patch = False


def _patched_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    return _original_linear(input, weight, bias)


def patch_linear() -> None:
    global _original_linear
    if _original_linear is not None:
        return
    _original_linear = F.linear
    F.linear = _patched_linear
    try:
        import torch.nn.functional as _F_alias
        _F_alias.linear = _patched_linear
    except Exception:
        pass


def unpatch_linear() -> None:
    global _original_linear
    if _original_linear is not None:
        F.linear = _original_linear
        try:
            import torch.nn.functional as _F_alias
            _F_alias.linear = _original_linear
        except Exception:
            pass
        _original_linear = None


def hvp_available() -> bool:
    """Return True if the Triton HVP kernel loaded successfully."""
    return _HVP_AVAILABLE
