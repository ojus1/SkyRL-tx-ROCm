from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.qwen3_5_gdn_superblock import (
    Qwen35GDNSuperblockConfig,
    qwen35_gdn_superblock_logical_buffers,
    qwen35_gdn_superblocks,
)
from skyrl.tx.models.qwen3_5 import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)


def _inputs(
    sequence: int,
    *,
    batch: int = 1,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, ...]:
    key_heads, value_heads, key_dim, value_dim = 2, 4, 4, 3
    keys = jax.random.split(jax.random.key(1700 + sequence + batch), 6)
    query = (
        jax.random.normal(
            keys[0], (batch, sequence, key_heads, key_dim), dtype=jnp.float32
        )
        * 0.2
    ).astype(dtype)
    key = (jax.random.normal(keys[1], query.shape, dtype=jnp.float32) * 0.2).astype(
        dtype
    )
    value = (
        jax.random.normal(
            keys[2], (batch, sequence, value_heads, value_dim), dtype=jnp.float32
        )
        * 0.2
    ).astype(dtype)
    g = -jax.random.uniform(
        keys[3],
        (batch, sequence, value_heads),
        dtype=jnp.float32,
        minval=0.001,
        maxval=0.02,
    )
    beta = jax.random.uniform(
        keys[4],
        g.shape,
        dtype=jnp.float32,
        minval=0.05,
        maxval=0.3,
    ).astype(dtype)
    initial_state = (
        jax.random.normal(
            keys[5],
            (batch, value_heads, key_dim, value_dim),
            dtype=jnp.float32,
        )
        * 0.02
    )
    return query, key, value, g, beta, initial_state


def _masked_reference(
    implementation: Callable[..., tuple[jax.Array, jax.Array]],
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
    attention_mask: jax.Array,
    *,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    repeats = value.shape[2] // query.shape[2]
    mask = attention_mask[..., None]
    repeated_query = jnp.repeat(query * mask[..., None], repeats, axis=2)
    repeated_key = jnp.repeat(key * mask[..., None], repeats, axis=2)
    masked_value = value * mask[..., None]
    masked_g = g * mask
    masked_beta = beta * mask
    if implementation is chunk_gated_delta_rule:
        return implementation(
            repeated_query,
            repeated_key,
            masked_value,
            masked_g,
            masked_beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
        )
    return implementation(
        repeated_query,
        repeated_key,
        masked_value,
        masked_g,
        masked_beta,
        initial_state=initial_state,
    )


@pytest.mark.parametrize(
    ("chunks_per_superblock", "sequence", "valid_lengths"),
    [
        (8, 511, (511, 509)),
        (8, 512, (512, 509)),
        (8, 513, (513, 509)),
        (16, 1025, (1025, 1021)),
    ],
)
def test_forward_matches_chunk_and_recurrent_at_real_superblock_boundaries(
    chunks_per_superblock: int,
    sequence: int,
    valid_lengths: tuple[int, int],
) -> None:
    inputs = _inputs(sequence, batch=2)
    *core_inputs, initial_state = inputs
    attention_mask = jnp.arange(sequence)[None, :] < jnp.asarray(valid_lengths)[:, None]
    config = Qwen35GDNSuperblockConfig(
        chunk_size=64,
        chunks_per_superblock=chunks_per_superblock,
    )

    actual = jax.jit(
        lambda *values: qwen35_gdn_superblocks(
            *values,
            attention_mask=attention_mask,
            initial_state=initial_state,
            config=config,
        )
    )(*core_inputs)
    chunk_expected = jax.jit(
        lambda *values: _masked_reference(
            chunk_gated_delta_rule,
            *values,
            initial_state,
            attention_mask,
            chunk_size=64,
        )
    )(*core_inputs)
    recurrent_expected = jax.jit(
        lambda *values: _masked_reference(
            recurrent_gated_delta_rule,
            *values,
            initial_state,
            attention_mask,
            chunk_size=64,
        )
    )(*core_inputs)

    assert actual[0].shape == (2, sequence, 4, 3)
    assert actual[1].shape == (2, 4, 4, 3)
    assert actual[0].dtype == jnp.float32
    assert actual[1].dtype == jnp.float32
    for expected in (chunk_expected, recurrent_expected):
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=2e-6)

    # The second sequence is right-padded across either a chunk or superblock
    # boundary.  Masked queries must produce zero output exactly.
    np.testing.assert_array_equal(
        np.asarray(actual[0][1, valid_lengths[1] :]),
        np.zeros_like(np.asarray(actual[0][1, valid_lengths[1] :])),
    )


@pytest.mark.parametrize(
    ("chunks_per_superblock", "sequence", "valid_length"),
    [(8, 513, 509), (16, 1025, 1021)],
)
def test_vjp_matches_chunk_and_recurrent_with_reverse_superblock_recomputation(
    chunks_per_superblock: int,
    sequence: int,
    valid_length: int,
) -> None:
    inputs = _inputs(sequence)
    attention_mask = jnp.arange(sequence)[None, :] < valid_length
    config = Qwen35GDNSuperblockConfig(
        chunk_size=64,
        chunks_per_superblock=chunks_per_superblock,
    )
    output_key, state_key = jax.random.split(
        jax.random.key(9100 + sequence + chunks_per_superblock)
    )
    output_weight = jax.random.normal(
        output_key,
        (1, sequence, 4, 3),
        dtype=jnp.float32,
    )
    state_weight = jax.random.normal(
        state_key,
        (1, 4, 4, 3),
        dtype=jnp.float32,
    )

    def loss(
        implementation: str,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
    ) -> jax.Array:
        if implementation == "superblock":
            output, final_state = qwen35_gdn_superblocks(
                query,
                key,
                value,
                g,
                beta,
                attention_mask=attention_mask,
                initial_state=initial_state,
                config=config,
            )
        else:
            reference = (
                chunk_gated_delta_rule
                if implementation == "chunk"
                else recurrent_gated_delta_rule
            )
            output, final_state = _masked_reference(
                reference,
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                attention_mask,
                chunk_size=64,
            )
        return jnp.sum(output.astype(jnp.float32) * output_weight) + jnp.sum(
            final_state * state_weight
        )

    results = {}
    for implementation in ("superblock", "chunk", "recurrent"):
        results[implementation] = jax.jit(
            jax.value_and_grad(
                lambda *values: loss(implementation, *values),
                argnums=tuple(range(len(inputs))),
            )
        )(*inputs)

    actual_value, actual_gradients = results["superblock"]
    for implementation in ("chunk", "recurrent"):
        expected_value, expected_gradients = results[implementation]
        np.testing.assert_allclose(actual_value, expected_value, rtol=2e-5, atol=2e-6)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)


def test_interior_left_and_all_masked_tokens_are_identity_transitions() -> None:
    sequence = 130
    inputs = _inputs(sequence)
    *core_inputs, initial_state = inputs
    masked_indices = jnp.asarray((0, 1, 17, 63, 64, 65, 97, 129))
    attention_mask = (
        jnp.ones((1, sequence), dtype=jnp.bool_).at[:, masked_indices].set(False)
    )
    config = Qwen35GDNSuperblockConfig(
        chunk_size=64,
        chunks_per_superblock=8,
    )

    actual = jax.jit(
        lambda *values: qwen35_gdn_superblocks(
            *values,
            attention_mask=attention_mask,
            initial_state=initial_state,
            config=config,
        )
    )(*core_inputs)
    for reference in (chunk_gated_delta_rule, recurrent_gated_delta_rule):
        expected = jax.jit(
            lambda *values: _masked_reference(
                reference,
                *values,
                initial_state,
                attention_mask,
                chunk_size=64,
            )
        )(*core_inputs)
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=2e-6)

    np.testing.assert_array_equal(
        np.asarray(actual[0][:, masked_indices]),
        np.zeros_like(np.asarray(actual[0][:, masked_indices])),
    )

    def masked_loss(*values: jax.Array) -> jax.Array:
        output, final_state = qwen35_gdn_superblocks(
            *values,
            attention_mask=attention_mask,
            initial_state=initial_state,
            config=config,
        )
        return jnp.sum(output.astype(jnp.float32)) + jnp.sum(final_state)

    gradients = jax.jit(jax.grad(masked_loss, argnums=tuple(range(len(core_inputs)))))(
        *core_inputs
    )
    for gradient in gradients:
        np.testing.assert_array_equal(
            np.asarray(gradient[:, masked_indices]),
            np.zeros_like(np.asarray(gradient[:, masked_indices])),
        )

    all_masked_output, all_masked_state = jax.jit(
        lambda *values: qwen35_gdn_superblocks(
            *values,
            attention_mask=jnp.zeros((1, sequence), dtype=jnp.bool_),
            initial_state=initial_state,
            config=config,
        )
    )(*core_inputs)
    np.testing.assert_array_equal(
        np.asarray(all_masked_output),
        np.zeros_like(np.asarray(all_masked_output)),
    )
    np.testing.assert_array_equal(
        np.asarray(all_masked_state), np.asarray(initial_state)
    )


def test_bf16_vjp_matches_current_chunk_rule_with_bounded_qk_drift() -> None:
    sequence = 129
    inputs = _inputs(sequence, dtype=jnp.bfloat16)
    attention_mask = jnp.arange(sequence)[None, :] < 125
    config = Qwen35GDNSuperblockConfig(
        chunk_size=64,
        chunks_per_superblock=8,
    )
    output_weight = jax.random.normal(
        jax.random.key(9200),
        (1, sequence, 4, 3),
        dtype=jnp.float32,
    )
    state_weight = jax.random.normal(
        jax.random.key(9201),
        (1, 4, 4, 3),
        dtype=jnp.float32,
    )

    def loss(implementation: str, *values: jax.Array) -> jax.Array:
        *core_inputs, initial_state = values
        if implementation == "superblock":
            output, final_state = qwen35_gdn_superblocks(
                *core_inputs,
                attention_mask=attention_mask,
                initial_state=initial_state,
                config=config,
            )
        else:
            output, final_state = _masked_reference(
                chunk_gated_delta_rule,
                *core_inputs,
                initial_state,
                attention_mask,
                chunk_size=64,
            )
        return jnp.sum(output.astype(jnp.float32) * output_weight) + jnp.sum(
            final_state * state_weight
        )

    results = {}
    for implementation in ("superblock", "chunk"):
        results[implementation] = jax.jit(
            jax.value_and_grad(
                lambda *values: loss(implementation, *values),
                argnums=tuple(range(len(inputs))),
            )
        )(*inputs)

    actual_value, actual_gradients = results["superblock"]
    expected_value, expected_gradients = results["chunk"]
    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-2, atol=2e-2)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        actual_f32 = np.asarray(actual, dtype=np.float32)
        expected_f32 = np.asarray(expected, dtype=np.float32)
        relative_l2 = np.linalg.norm(actual_f32 - expected_f32) / max(
            np.linalg.norm(expected_f32),
            1e-12,
        )
        assert relative_l2 < 0.02


def test_bf16_output_and_fp32_state_match_current_chunk_contract() -> None:
    sequence = 513
    inputs = _inputs(sequence, dtype=jnp.bfloat16)
    *core_inputs, initial_state = inputs
    attention_mask = jnp.arange(sequence)[None, :] < 509
    config = Qwen35GDNSuperblockConfig(
        chunk_size=64,
        chunks_per_superblock=8,
    )

    actual = jax.jit(
        lambda *values: qwen35_gdn_superblocks(
            *values,
            attention_mask=attention_mask,
            initial_state=initial_state,
            config=config,
        )
    )(*core_inputs)
    expected = jax.jit(
        lambda *values: _masked_reference(
            chunk_gated_delta_rule,
            *values,
            initial_state,
            attention_mask,
            chunk_size=64,
        )
    )(*core_inputs)

    assert actual[0].dtype == jnp.bfloat16
    assert actual[1].dtype == jnp.float32
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-2, atol=2e-2)


def test_exact_qwen35_logical_buffer_accounting() -> None:
    buffers_512 = qwen35_gdn_superblock_logical_buffers(
        batch_size=1,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        config=Qwen35GDNSuperblockConfig(
            chunk_size=64,
            chunks_per_superblock=8,
        ),
    )
    buffers_1024 = qwen35_gdn_superblock_logical_buffers(
        batch_size=1,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        config=Qwen35GDNSuperblockConfig(),
    )

    assert buffers_512["boundary_state"] == 2 * 1024**2
    assert buffers_1024["boundary_state"] == 2 * 1024**2
    assert buffers_512["u_plus_w"] == 16 * 1024**2
    assert buffers_1024["u_plus_w"] == 32 * 1024**2
    assert buffers_512["decay_mask"] == 4 * 1024**2
    assert buffers_1024["decay_mask"] == 8 * 1024**2
    assert buffers_1024["rhs"] == 32 * 1024**2
    assert buffers_1024["solution"] == 32 * 1024**2


def test_shape_and_schedule_validation() -> None:
    with pytest.raises(ValueError, match="chunks_per_superblock"):
        Qwen35GDNSuperblockConfig(chunks_per_superblock=4)
    with pytest.raises(ValueError, match="chunk_size"):
        Qwen35GDNSuperblockConfig(chunk_size=0)

    query, key, value, g, beta, initial_state = _inputs(65)
    with pytest.raises(ValueError, match="divisible"):
        qwen35_gdn_superblocks(
            query,
            key,
            value[:, :, :3],
            g[:, :, :3],
            beta[:, :, :3],
            initial_state=initial_state[:, :3],
            config=Qwen35GDNSuperblockConfig(chunk_size=8),
        )
    with pytest.raises(ValueError, match=r"\[B,T\]"):
        qwen35_gdn_superblocks(
            query,
            key,
            value,
            g,
            beta,
            attention_mask=jnp.ones((1, 64), dtype=jnp.bool_),
            initial_state=initial_state,
            config=Qwen35GDNSuperblockConfig(chunk_size=8),
        )
