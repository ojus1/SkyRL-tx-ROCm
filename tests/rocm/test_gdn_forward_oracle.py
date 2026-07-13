from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from skyrl.tx.kernels.rocm import gdn_forward_oracle as oracle

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORACLE_SOURCE = (
    _REPO_ROOT / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_forward_oracle.py"
)


def _inputs(
    sequence: int,
    *,
    key_heads: int,
    value_heads: int,
    key_dimension: int,
    value_dimension: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    query = (rng.standard_normal((1, sequence, key_heads, key_dimension)) * 0.3).astype(
        np.float32
    )
    key = (rng.standard_normal(query.shape) * 0.3).astype(np.float32)
    value = (
        rng.standard_normal((1, sequence, value_heads, value_dimension)) * 0.1
    ).astype(np.float32)
    g = -rng.uniform(0.001, 0.01, (1, sequence, value_heads)).astype(np.float32)
    beta = rng.uniform(0.01, 0.12, g.shape).astype(np.float32)
    initial_state = (
        rng.standard_normal((1, value_heads, key_dimension, value_dimension)) * 0.01
    ).astype(np.float32)
    return query, key, value, g, beta, initial_state


def _assert_forward_close(
    actual: tuple[np.ndarray, np.ndarray], expected: tuple[np.ndarray, np.ndarray]
) -> None:
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-5, atol=1e-7)


def test_provenance_is_pinned_and_default_oracle_is_numpy_only() -> None:
    source = _ORACLE_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0] for node in imports for alias in node.names
    }

    assert oracle.FLASHQLA_REVISION == "40b7527f6c6e2ed8ed65350103e3ca64174f53f3"
    assert oracle.FLASHQLA_REVISION in oracle.FLASHQLA_KKT_SOURCE
    assert oracle.FLASHQLA_REVISION in oracle.FLASHQLA_FUSED_FORWARD_SOURCE
    assert oracle.SKYRL_GDN_EQUATION_SOURCE == "skyrl/tx/models/qwen3_5.py"
    assert imported_roots.isdisjoint({"jax", "jaxlib", "torch", "tilelang"})
    assert "np.linalg.solve" in source
    assert "finite-difference estimates" in source


def test_flashqla_kkt_and_skyrl_decay_folded_representations_are_conjugate() -> None:
    rng = np.random.default_rng(2035)
    key = (rng.standard_normal((2, 7, 5)) * 0.2).astype(np.float32)
    value = (rng.standard_normal((2, 7, 3)) * 0.2).astype(np.float32)
    g = -rng.uniform(0.002, 0.02, (2, 7)).astype(np.float32)
    beta = rng.uniform(0.03, 0.25, (2, 7)).astype(np.float32)

    flashqla = oracle.flashqla_chunk_representation_numpy(key, value, g, beta)
    skyrl = oracle.skyrl_wy_chunk_representation_numpy(key, value, g, beta)

    conjugated_inverse = flashqla.decay * flashqla.basis_inverse
    np.testing.assert_allclose(
        skyrl.basis_inverse, conjugated_inverse, rtol=3e-6, atol=3e-7
    )
    np.testing.assert_allclose(
        skyrl.effective_update, flashqla.effective_update, rtol=3e-6, atol=3e-7
    )
    np.testing.assert_allclose(
        skyrl.prepared_u, flashqla.prepared_u, rtol=3e-6, atol=3e-7
    )
    np.testing.assert_allclose(
        skyrl.prepared_w, flashqla.prepared_w, rtol=3e-6, atol=3e-7
    )


@pytest.mark.parametrize("sequence", [64, 128])
def test_exact_qwen35_geometry_matches_recurrence_with_nonzero_state(
    sequence: int,
) -> None:
    *inputs, initial_state = _inputs(
        sequence,
        key_heads=16,
        value_heads=32,
        key_dimension=128,
        value_dimension=128,
        seed=3500 + sequence,
    )
    expected = oracle.recurrent_gdn_forward_numpy(*inputs, initial_state=initial_state)
    flashqla = oracle.flashqla_gdn_forward_numpy(*inputs, initial_state=initial_state)
    skyrl = oracle.skyrl_wy_gdn_forward_numpy(*inputs, initial_state=initial_state)

    assert flashqla[0].shape == (1, sequence, 32, 128)
    assert flashqla[1].shape == (1, 32, 128, 128)
    assert all(
        array.dtype == np.float32
        for result in (expected, flashqla, skyrl)
        for array in result
    )
    _assert_forward_close(flashqla, expected)
    _assert_forward_close(skyrl, expected)
    _assert_forward_close(flashqla, skyrl)


@pytest.mark.parametrize("valid_length", [63, 64, 65])
def test_right_padding_at_chunk_boundary_is_zero_output_identity_transition(
    valid_length: int,
) -> None:
    *inputs, initial_state = _inputs(
        128,
        key_heads=2,
        value_heads=4,
        key_dimension=8,
        value_dimension=7,
        seed=6400 + valid_length,
    )
    mask = np.arange(128)[None, :] < valid_length
    expected = oracle.recurrent_gdn_forward_numpy(
        *inputs, attention_mask=mask, initial_state=initial_state
    )

    for implementation in (
        oracle.flashqla_gdn_forward_numpy,
        oracle.skyrl_wy_gdn_forward_numpy,
    ):
        actual = implementation(
            *inputs, attention_mask=mask, initial_state=initial_state
        )
        _assert_forward_close(actual, expected)
        np.testing.assert_array_equal(
            actual[0][:, valid_length:], np.zeros_like(actual[0][:, valid_length:])
        )

        prefix_inputs = tuple(array[:, :valid_length].copy() for array in inputs)
        prefix = implementation(*prefix_inputs, initial_state=initial_state)
        np.testing.assert_allclose(
            actual[0][:, :valid_length], prefix[0], rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(actual[1], prefix[1], rtol=1e-5, atol=1e-7)


def test_grouped_head_mapping_uses_value_head_integer_divided_by_two() -> None:
    query = np.zeros((1, 64, 2, 2), dtype=np.float32)
    key = np.zeros_like(query)
    value = np.zeros((1, 64, 4, 1), dtype=np.float32)
    g = np.zeros((1, 64, 4), dtype=np.float32)
    beta = np.zeros_like(g)
    mask = np.zeros((1, 64), dtype=np.bool_)
    mask[:, 0] = True
    query[0, 0] = np.eye(2, dtype=np.float32)
    key[0, 0] = np.eye(2, dtype=np.float32)
    value[0, 0, :, 0] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    beta[0, 0] = 1.0
    expected_first = value[0, 0, :, 0] * np.float32(1.0 / math.sqrt(2.0))

    for implementation in (
        oracle.recurrent_gdn_forward_numpy,
        oracle.flashqla_gdn_forward_numpy,
        oracle.skyrl_wy_gdn_forward_numpy,
    ):
        output, _ = implementation(query, key, value, g, beta, attention_mask=mask)
        np.testing.assert_allclose(
            output[0, 0, :, 0], expected_first, rtol=2e-6, atol=2e-6
        )
        np.testing.assert_array_equal(output[:, 1:], np.zeros_like(output[:, 1:]))


def test_all_masked_tokens_leave_nonzero_state_exactly_unchanged() -> None:
    *inputs, initial_state = _inputs(
        64,
        key_heads=2,
        value_heads=4,
        key_dimension=5,
        value_dimension=3,
        seed=9917,
    )
    mask = np.zeros((1, 64), dtype=np.bool_)
    for implementation in (
        oracle.recurrent_gdn_forward_numpy,
        oracle.flashqla_gdn_forward_numpy,
        oracle.skyrl_wy_gdn_forward_numpy,
    ):
        output, final_state = implementation(
            *inputs, attention_mask=mask, initial_state=initial_state
        )
        np.testing.assert_array_equal(output, np.zeros_like(output))
        np.testing.assert_array_equal(final_state, initial_state)


def test_public_oracle_rejects_non_fp32_and_noncanonical_state() -> None:
    *inputs, initial_state = _inputs(
        64,
        key_heads=2,
        value_heads=4,
        key_dimension=5,
        value_dimension=3,
        seed=1205,
    )
    with pytest.raises(TypeError, match="float32"):
        oracle.flashqla_gdn_forward_numpy(inputs[0].astype(np.float64), *inputs[1:])
    with pytest.raises(ValueError, match="initial_state"):
        oracle.skyrl_wy_gdn_forward_numpy(
            *inputs, initial_state=initial_state[:, :, :, :2]
        )
    with pytest.raises(ValueError, match="chunk_size=64"):
        oracle.flashqla_gdn_forward_numpy(*inputs, chunk_size=32)
