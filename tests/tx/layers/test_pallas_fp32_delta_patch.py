import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas.ops.gpu import attention as jax_pallas_attention

from skyrl.tx.layers import attention

_ORIGINAL_PREPROCESS_BACKWARD_KERNEL = jax_pallas_attention._preprocess_backward_kernel
_GRADIENT_RELATIVE_L2_GATE = 0.03


@pytest.fixture
def unpatched_preprocess_kernel():
    """Keep the process-global JAX private function isolated between tests."""
    jax_pallas_attention._preprocess_backward_kernel = (
        _ORIGINAL_PREPROCESS_BACKWARD_KERNEL
    )
    try:
        yield
    finally:
        jax_pallas_attention._preprocess_backward_kernel = (
            _ORIGINAL_PREPROCESS_BACKWARD_KERNEL
        )


def test_fp32_delta_patch_installation_is_guarded_and_idempotent(
    unpatched_preprocess_kernel,
):
    assert attention._install_pallas_fp32_delta_preprocess_patch(jax_pallas_attention)
    assert (
        jax_pallas_attention._preprocess_backward_kernel
        is attention._pallas_fp32_delta_preprocess_kernel
    )
    assert not attention._install_pallas_fp32_delta_preprocess_patch(
        jax_pallas_attention
    )


def test_fp32_delta_patch_rejects_a_different_jax_version(
    monkeypatch, unpatched_preprocess_kernel
):
    monkeypatch.setattr(attention.jax, "__version__", "0.10.3")

    with pytest.raises(RuntimeError, match=r"requires exactly JAX 0\.10\.2"):
        attention._install_pallas_fp32_delta_preprocess_patch(jax_pallas_attention)

    assert (
        jax_pallas_attention._preprocess_backward_kernel
        is _ORIGINAL_PREPROCESS_BACKWARD_KERNEL
    )


def test_fp32_delta_patch_rejects_private_source_drift(unpatched_preprocess_kernel):
    def changed_preprocess(out_ref, dout_ref, delta_ref, head_dim: int):
        del out_ref, dout_ref, delta_ref, head_dim

    jax_pallas_attention._preprocess_backward_kernel = changed_preprocess

    with pytest.raises(RuntimeError, match="preprocess source"):
        attention._install_pallas_fp32_delta_preprocess_patch(jax_pallas_attention)

    assert jax_pallas_attention._preprocess_backward_kernel is changed_preprocess


def test_pallas_mha_installs_fp32_delta_patch_before_call(
    monkeypatch, unpatched_preprocess_kernel
):
    expected = object()

    def fake_mha(*args, **kwargs):
        assert (
            jax_pallas_attention._preprocess_backward_kernel
            is attention._pallas_fp32_delta_preprocess_kernel
        )
        return expected

    fake_mha.__signature__ = inspect.signature(
        lambda q, k, v, segment_ids, *, block_sizes, backward_pass_impl, num_warps, num_stages: (
            None
        )
    )
    monkeypatch.setattr(jax_pallas_attention, "mha", fake_mha)

    result = attention._pallas_mha(object(), object(), object(), object(), 0.0625)

    assert result is expected


def _chunked_fp32_reference(q, k, v, mask, scale, *, query_block_size=16):
    """Match the bounded FP32 oracle used by the isolated attention probe."""
    batch_size, sequence_length, query_heads, head_dim = q.shape
    repeats = query_heads // k.shape[2]
    k_repeated = jnp.repeat(k, repeats, axis=2).astype(jnp.float32)
    v_repeated = jnp.repeat(v, repeats, axis=2).astype(jnp.float32)
    query_blocks = q.reshape(
        batch_size,
        sequence_length // query_block_size,
        query_block_size,
        query_heads,
        head_dim,
    ).transpose(1, 0, 2, 3, 4)
    query_positions = jnp.arange(sequence_length, dtype=jnp.int32).reshape(
        sequence_length // query_block_size, query_block_size
    )
    key_positions = jnp.arange(sequence_length, dtype=jnp.int32)
    valid_keys = mask.astype(bool)[:, None, None, :]

    def attend_query_block(items):
        q_block, q_positions = items
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk",
            q_block.astype(jnp.float32),
            k_repeated,
            preferred_element_type=jnp.float32,
        )
        logits *= scale
        causal = key_positions[None, None, None, :] <= q_positions[None, None, :, None]
        logits = jnp.where(valid_keys & causal, logits, jnp.finfo(jnp.float32).min)
        probabilities = jax.nn.softmax(logits, axis=-1)
        output = jnp.einsum(
            "bhqk,bkhd->bqhd",
            probabilities,
            v_repeated,
            preferred_element_type=jnp.float32,
        )
        return output.astype(q.dtype)

    output_blocks = jax.lax.map(attend_query_block, (query_blocks, query_positions))
    return output_blocks.transpose(1, 0, 2, 3, 4).reshape(q.shape)


def _relative_l2(actual, expected) -> float:
    actual_fp32 = actual.astype(jnp.float32)
    expected_fp32 = expected.astype(jnp.float32)
    return float(
        jnp.linalg.norm((actual_fp32 - expected_fp32).ravel())
        / jnp.linalg.norm(expected_fp32.ravel())
    )


def test_fp32_delta_patch_improves_exact_qwen35_bf16_interpret_gradients(
    unpatched_preprocess_kernel,
):
    """Exercise Qwen3.5's exact Hq/Hkv/D geometry without dispatching a GPU."""
    sequence_length, query_heads, kv_heads, head_dim = 64, 16, 4, 256
    repeats = query_heads // kv_heads
    scale = head_dim**-0.5
    cpu = jax.devices("cpu")[0]

    with jax.default_device(cpu):
        q_key, k_key, v_key = jax.random.split(jax.random.key(0), 3)
        q = jax.random.normal(
            q_key, (1, sequence_length, query_heads, head_dim), dtype=jnp.bfloat16
        )
        k = jax.random.normal(
            k_key, (1, sequence_length, kv_heads, head_dim), dtype=jnp.bfloat16
        )
        v = jax.random.normal(
            v_key, (1, sequence_length, kv_heads, head_dim), dtype=jnp.bfloat16
        )
        segment_ids = jnp.ones((1, sequence_length), dtype=jnp.int32)
        valid_query_mask = segment_ids[:, :, None, None].astype(jnp.float32)
        k_repeated = jnp.repeat(k, repeats, axis=2)
        v_repeated = jnp.repeat(v, repeats, axis=2)
        block_sizes = jax_pallas_attention.BlockSizes(64, 64, 32, 32, 32, 32)

        def reference(q_arg, k_arg, v_arg):
            return _chunked_fp32_reference(q_arg, k_arg, v_arg, segment_ids, scale)

        def pallas(q_arg, k_arg, v_arg):
            return jax_pallas_attention.mha(
                q_arg,
                k_arg,
                v_arg,
                segment_ids,
                sm_scale=scale,
                causal=True,
                block_sizes=block_sizes,
                backward_pass_impl="triton",
                num_warps=4,
                num_stages=1,
                interpret=True,
            )

        def loss(function, *args):
            output = function(*args).astype(jnp.float32)
            return jnp.sum(output**2 * valid_query_mask)

        reference_gradients = jax.grad(
            lambda *args: loss(reference, *args), argnums=(0, 1, 2)
        )(q, k, v)

        def pallas_gradients():
            expanded_gradients = jax.grad(
                lambda *args: loss(pallas, *args), argnums=(0, 1, 2)
            )(q, k_repeated, v_repeated)
            return (
                expanded_gradients[0],
                expanded_gradients[1]
                .reshape(1, sequence_length, kv_heads, repeats, head_dim)
                .sum(axis=3),
                expanded_gradients[2]
                .reshape(1, sequence_length, kv_heads, repeats, head_dim)
                .sum(axis=3),
            )

        original_output = pallas(q, k_repeated, v_repeated)
        original_gradients = pallas_gradients()
        original_errors = tuple(
            _relative_l2(actual, expected)
            for actual, expected in zip(
                original_gradients, reference_gradients, strict=True
            )
        )

        assert attention._install_pallas_fp32_delta_preprocess_patch(
            jax_pallas_attention
        )
        patched_output = pallas(q, k_repeated, v_repeated)
        patched_gradients = pallas_gradients()
        patched_errors = tuple(
            _relative_l2(actual, expected)
            for actual, expected in zip(
                patched_gradients, reference_gradients, strict=True
            )
        )
        reference_gradients_after_install = jax.grad(
            lambda *args: loss(reference, *args), argnums=(0, 1, 2)
        )(q, k, v)

    np.testing.assert_array_equal(
        np.asarray(patched_output), np.asarray(original_output)
    )
    np.testing.assert_array_equal(
        np.asarray(patched_gradients[2]), np.asarray(original_gradients[2])
    )
    for before, after in zip(
        reference_gradients, reference_gradients_after_install, strict=True
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))

    # The accepted Pallas-attention gradient ceiling is 3%.  Independently
    # require a material dQ/dK improvement so the relaxed absolute ceiling
    # cannot promote a patch that merely stays within tolerance.
    for index in (0, 1):
        assert patched_errors[index] < _GRADIENT_RELATIVE_L2_GATE
        assert patched_errors[index] < original_errors[index] * 0.65
