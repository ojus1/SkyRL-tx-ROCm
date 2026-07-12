"""Shared attention utilities for transformer models."""

import hashlib
import inspect
import os

import jax
import jax.numpy as jnp
from jax.extend import backend as jax_backend

# cuDNN flash attention supported dtypes
# https://github.com/jax-ml/jax/blob/8b1f782540f71fbe230a2dccd331975faafc6c83/jax/_src/cudnn/fused_attention_stablehlo.py#L290
_CUDNN_SUPPORTED_DTYPES = (
    jnp.float16,
    jnp.bfloat16,
    jnp.float8_e4m3fn,
    jnp.float8_e5m2,
)

# The Pallas Triton attention kernel is experimental on ROCm, particularly its
# backward pass. Keep it opt-in until the target GPU has passed the forward,
# backward, padding, and long-context stress matrix.
_ROCM_PALLAS_ATTENTION_ENV = "SKYRL_ROCM_PALLAS_ATTENTION"
_PALLAS_SUPPORTED_DTYPES = (jnp.float16, jnp.bfloat16)
_PALLAS_SUPPORTED_HEAD_DIMS = (256,)
_PALLAS_SUPPORTED_Q_HEADS = 16
_PALLAS_SUPPORTED_KV_HEADS = 4
_PALLAS_FORWARD_BLOCK_SIZE = 64
_PALLAS_BACKWARD_BLOCK_SIZE = 32
_PALLAS_REQUIRED_SEQ_LEN = 512
# JAX 0.10.2's Pallas MHA computes the backward softmax delta from BF16
# ``out`` and ``dout`` loads. Promoting just those two tiles before their
# product/reduction materially improves Qwen3.5 dQ/dK accuracy without
# changing the kernel's mask, output shape, or store contract. This is a
# deliberately narrow private-API patch: any JAX/source drift must be reviewed
# instead of silently accepting an unvalidated implementation.
_PALLAS_FP32_DELTA_JAX_VERSION = "0.10.2"
_PALLAS_PREPROCESS_BACKWARD_KERNEL_PARAMETERS = (
    "out_ref",
    "dout_ref",
    "delta_ref",
    "head_dim",
)
_PALLAS_PREPROCESS_BACKWARD_KERNEL_SHA256 = (
    "db93e4c600257e7cd149301a5fc71b5444d74498283bc870e3e03485861c26ac"
)
# A monolithic 32K backward launch wedged the gfx1100 graphics ring. Keep the
# validated kernel below that watchdog boundary until query-range chunking
# makes each launch independently bounded.
_PALLAS_MAX_SEQ_LEN = 16_384


def _has_cuda_backend() -> bool:
    """Return whether JAX's generic GPU platform is backed by CUDA, not ROCm."""
    if jax.default_backend() != "gpu":
        return False
    return "rocm" not in jax_backend.get_backend().platform_version.lower()


def _has_rocm_backend() -> bool:
    """Return whether JAX's generic GPU platform is backed by ROCm."""
    if jax.default_backend() != "gpu":
        return False
    return "rocm" in jax_backend.get_backend().platform_version.lower()


def _rocm_pallas_attention_enabled() -> bool:
    """Return whether the experimental ROCm Pallas attention path is enabled."""
    value = os.environ.get(_ROCM_PALLAS_ATTENTION_ENV, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {_ROCM_PALLAS_ATTENTION_ENV}={value!r}; expected one of "
        "1/true/yes/on or 0/false/no/off"
    )


def _can_use_rocm_pallas_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    is_causal: bool,
    head_dim: int,
) -> bool:
    """Return whether inputs satisfy the conservative ROCm Pallas contract."""
    if not (_has_rocm_backend() and _rocm_pallas_attention_enabled() and is_causal):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or attention_mask.ndim != 2:
        return False
    if (
        q.dtype not in _PALLAS_SUPPORTED_DTYPES
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        return False
    if not (
        jnp.issubdtype(attention_mask.dtype, jnp.bool_)
        or jnp.issubdtype(attention_mask.dtype, jnp.integer)
    ):
        return False

    batch_size, q_len, num_q_heads, q_head_dim = q.shape
    kv_batch_size, kv_len, num_kv_heads, k_head_dim = k.shape
    v_batch_size, v_len, num_v_heads, v_head_dim = v.shape
    if batch_size != 1 or q_len <= 0:
        return False
    if batch_size != kv_batch_size or batch_size != v_batch_size:
        return False
    if q_len != kv_len or q_len != v_len or attention_mask.shape != (batch_size, q_len):
        return False
    if num_kv_heads != num_v_heads or num_q_heads % num_kv_heads != 0:
        return False
    if (
        num_q_heads != _PALLAS_SUPPORTED_Q_HEADS
        or num_kv_heads != _PALLAS_SUPPORTED_KV_HEADS
    ):
        return False
    if q_head_dim != head_dim or k_head_dim != head_dim or v_head_dim != head_dim:
        return False
    if head_dim not in _PALLAS_SUPPORTED_HEAD_DIMS:
        return False

    # Both the forward and fused backward Pallas block specs require exact
    # divisibility. A 64-token forward tile is intentionally conservative for
    # RDNA3's resource limits, while the installed backward kernel uses 32.
    return q_len % _PALLAS_FORWARD_BLOCK_SIZE == 0 and q_len <= _PALLAS_MAX_SEQ_LEN


def _is_large_causal_self_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, is_causal: bool
) -> bool:
    """Return whether a generic ROCm fallback would be a long quadratic path."""
    return (
        is_causal
        and q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.shape[1] == k.shape[1] == v.shape[1]
        and q.shape[1] >= _PALLAS_REQUIRED_SEQ_LEN
    )


def _pallas_fp32_delta_preprocess_kernel(out_ref, dout_ref, delta_ref, head_dim: int):
    """JAX 0.10.2 MHA delta preprocess with FP32 product/reduction inputs."""
    # Keep this load mask and write-back contract identical to the guarded JAX
    # source. Only the two explicit FP32 casts are intentional differences.
    from jax.experimental.pallas import triton as plgpu

    head_mask = (jnp.arange(out_ref.shape[-1]) < head_dim)[None, :]
    o = plgpu.load(out_ref, mask=head_mask, other=0.0).astype(jnp.float32)
    do = plgpu.load(dout_ref, mask=head_mask, other=0.0).astype(jnp.float32)
    delta = jnp.sum(o * do, axis=1)
    delta_ref[...] = delta.astype(delta_ref.dtype)


def _install_pallas_fp32_delta_preprocess_patch(pallas_attention_module=None) -> bool:
    """Install the guarded JAX 0.10.2 FP32-delta patch before MHA tracing.

    Returns ``True`` for the first installation and ``False`` when the exact
    replacement is already installed. Version, signature, and source checks
    intentionally fail closed because this integration patches a private JAX
    function.
    """
    if jax.__version__ != _PALLAS_FP32_DELTA_JAX_VERSION:
        raise RuntimeError(
            "ROCm Pallas FP32-delta preprocessing requires exactly "
            f"JAX {_PALLAS_FP32_DELTA_JAX_VERSION}; found {jax.__version__}. "
            "Refusing to patch an unvalidated private API."
        )

    if pallas_attention_module is None:
        try:
            from jax.experimental.pallas.ops.gpu import (
                attention as pallas_attention_module,
            )
        except ImportError as error:
            raise RuntimeError(
                "ROCm Pallas FP32-delta preprocessing requires JAX's experimental GPU attention API."
            ) from error

    kernel_name = "_preprocess_backward_kernel"
    original_kernel = getattr(pallas_attention_module, kernel_name, None)
    if original_kernel is _pallas_fp32_delta_preprocess_kernel:
        return False
    if original_kernel is None or not callable(original_kernel):
        raise RuntimeError(
            "Incompatible JAX 0.10.2 Pallas attention internals: "
            f"missing callable {kernel_name}. Refusing the private-API patch."
        )

    signature = inspect.signature(original_kernel)
    parameter_names = tuple(signature.parameters)
    parameter_kinds = tuple(
        parameter.kind for parameter in signature.parameters.values()
    )
    positional_or_keyword = inspect.Parameter.POSITIONAL_OR_KEYWORD
    if parameter_names != _PALLAS_PREPROCESS_BACKWARD_KERNEL_PARAMETERS or any(
        kind is not positional_or_keyword for kind in parameter_kinds
    ):
        raise RuntimeError(
            "Incompatible JAX 0.10.2 Pallas attention preprocess signature; "
            "refusing the private-API patch."
        )

    try:
        source = inspect.getsource(original_kernel)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            "Could not inspect JAX 0.10.2 Pallas attention preprocess source; "
            "refusing the private-API patch."
        ) from error
    source_digest = hashlib.sha256(source.encode()).hexdigest()
    if source_digest != _PALLAS_PREPROCESS_BACKWARD_KERNEL_SHA256:
        raise RuntimeError(
            "Incompatible JAX 0.10.2 Pallas attention preprocess source "
            f"(sha256={source_digest}); refusing the private-API patch."
        )

    setattr(pallas_attention_module, kernel_name, _pallas_fp32_delta_preprocess_kernel)
    return True


def _pallas_mha(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array,
    scale: float,
) -> jax.Array:
    """Call JAX's Pallas Triton MHA with conservative ROCm tile settings."""
    try:
        from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha
    except ImportError as error:
        raise RuntimeError(
            "ROCm Pallas attention requires JAX's experimental GPU attention API; "
            "this integration was validated with JAX 0.10.2."
        ) from error

    # Install before ``mha`` is called: its custom VJP resolves this private
    # preprocess kernel while tracing the backward path.
    _install_pallas_fp32_delta_preprocess_patch()

    required_parameters = {
        "block_sizes",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
    }
    missing_parameters = required_parameters.difference(
        inspect.signature(mha).parameters
    )
    if missing_parameters:
        raise RuntimeError(
            "Incompatible JAX experimental Pallas attention API; missing parameters: "
            + ", ".join(sorted(missing_parameters))
        )

    block_sizes = BlockSizes(
        block_q=_PALLAS_FORWARD_BLOCK_SIZE,
        block_k=_PALLAS_FORWARD_BLOCK_SIZE,
        block_q_dkv=_PALLAS_BACKWARD_BLOCK_SIZE,
        block_kv_dkv=_PALLAS_BACKWARD_BLOCK_SIZE,
        block_q_dq=_PALLAS_BACKWARD_BLOCK_SIZE,
        block_kv_dq=_PALLAS_BACKWARD_BLOCK_SIZE,
    )
    return mha(
        q,
        k,
        v,
        segment_ids=segment_ids,
        sm_scale=scale,
        causal=True,
        block_sizes=block_sizes,
        backward_pass_impl="triton",
        num_warps=4,
        # This setting applies to the forward kernel. JAX 0.10.2's custom VJP
        # selects its own stage counts; both backward kernels are therefore
        # covered separately by the gfx1100 validation matrix.
        num_stages=1,
    )


def _rocm_pallas_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    scale: float,
) -> jax.Array:
    """Run causal Pallas attention after adapting GQA and right padding.

    The Pallas kernel accepts one segment ID per token rather than a key mask.
    For the documented right-padded causal input, casting the 1/0 mask creates
    one valid-token segment and one padding segment. Valid-token outputs match
    key masking; outputs at padded query positions may differ and must remain
    excluded by the training loss mask.
    """
    repeats = q.shape[2] // k.shape[2]
    if repeats != 1:
        # The installed Pallas MHA kernel does not implement grouped-query
        # attention. Repeat is differentiable, and its VJP sums each KV group's
        # gradients back into the original heads.
        k = jnp.repeat(k, repeats, axis=2)
        v = jnp.repeat(v, repeats, axis=2)
    segment_ids = (attention_mask != 0).astype(jnp.int32)
    return _pallas_mha(q, k, v, segment_ids, scale)


def dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_mask: jax.Array,
    is_causal: bool,
    head_dim: int,
) -> jax.Array:
    """Compute dot-product attention with automatic backend selection.

    Uses cuDNN on CUDA for memory-efficient attention. An experimental Pallas
    Triton path can be enabled on ROCm with ``SKYRL_ROCM_PALLAS_ATTENTION=1``;
    it is restricted to compatible causal prefill/training inputs. ROCm causal
    self-attention at 512 tokens or longer is never allowed to use quadratic
    XLA attention; shorter or non-causal inputs retain the portable fallback.

    Args:
        q: Query tensor of shape [batch, q_len, num_heads, head_dim]
        k: Key tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        v: Value tensor of shape [batch, kv_len, num_kv_heads, head_dim]
        attention_mask: Mask of shape [batch, kv_len] where 1 = valid, 0 = masked.
            Sequences must be right-padded (valid tokens first, then padding).
        is_causal: Whether to apply causal masking (for prefill/training)
        head_dim: Dimension of each attention head (for scaling)

    Returns:
        Attention output of shape [batch, q_len, num_heads, head_dim]
    """
    scale = 1.0 / head_dim**0.5

    if _has_cuda_backend() and q.dtype in _CUDNN_SUPPORTED_DTYPES:
        kv_seq_lengths = attention_mask.sum(axis=1).astype(jnp.int32)
        q_seq_lengths = jnp.minimum(kv_seq_lengths, q.shape[1])
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            is_causal=is_causal,
            query_seq_lengths=q_seq_lengths,
            key_value_seq_lengths=kv_seq_lengths,
            implementation="cudnn",
        )

    is_rocm = _has_rocm_backend()
    is_large_causal_self_attention = _is_large_causal_self_attention(q, k, v, is_causal)
    if is_rocm and is_large_causal_self_attention and q.shape[1] > _PALLAS_MAX_SEQ_LEN:
        raise ValueError(
            f"ROCm causal self-attention sequence length {q.shape[1]} exceeds the validated "
            f"{_PALLAS_MAX_SEQ_LEN}-token safety limit. A monolithic 32K backward launch "
            "caused a gfx1100 ring timeout; use a watchdog-safe chunked implementation."
        )

    if _can_use_rocm_pallas_attention(q, k, v, attention_mask, is_causal, head_dim):
        return _rocm_pallas_attention(q, k, v, attention_mask, scale)

    if is_rocm and is_large_causal_self_attention:
        if not _rocm_pallas_attention_enabled():
            raise ValueError(
                f"ROCm causal self-attention at {q.shape[1]} tokens cannot use the quadratic XLA "
                f"fallback. Explicitly enable the validated Pallas path with "
                f"{_ROCM_PALLAS_ATTENTION_ENV}=1, or use a shorter sequence."
            )
        raise ValueError(
            "ROCm Pallas attention was explicitly enabled, but this long causal self-attention input "
            "does not satisfy its dtype, head-shape, mask, or 64-token alignment contract. Refusing "
            "the quadratic XLA fallback; pad to a supported bucket."
        )

    # Portable CPU/TPU fallback, plus only bounded/non-causal ROCm inputs.
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        mask=attention_mask[:, None, None, :].astype(bool),
        is_causal=is_causal,
    )
