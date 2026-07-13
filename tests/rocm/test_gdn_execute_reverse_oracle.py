from __future__ import annotations

import ast
import hashlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from skyrl.tx.kernels.rocm import gdn_execute_reverse_oracle as reverse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORACLE_SOURCE = (
    _REPO_ROOT / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_execute_reverse_oracle.py"
)
_GRADIENT_NAMES = (
    "query",
    "key",
    "prepared_u",
    "prepared_w",
    "gamma",
    "initial_state",
)


def _exact_inputs() -> tuple[np.ndarray, ...]:
    return (
        np.zeros(reverse.GDN_EXECUTE_S512_QUERY_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_EXECUTE_S512_QUERY_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_EXECUTE_S512_PREPARED_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_EXECUTE_S512_PREPARED_SHAPE, dtype=np.float32),
        np.ones(reverse.GDN_EXECUTE_S512_GAMMA_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_EXECUTE_S512_STATE_SHAPE, dtype=np.float32),
        np.zeros(
            reverse.GDN_EXECUTE_S512_OUTPUT_SHAPE,
            dtype=reverse._bfloat16_dtype(),
        ),
        np.zeros(reverse.GDN_EXECUTE_S512_STATE_SHAPE, dtype=np.float32),
    )


def _hash_arrays(arrays: tuple[np.ndarray, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(array.view(np.uint8)).hexdigest() for array in arrays)


def _gradient_arrays(
    gradients: reverse.GDNExecuteBoundaryGradients,
) -> tuple[np.ndarray, ...]:
    return tuple(getattr(gradients, name) for name in _GRADIENT_NAMES)


def _reduced_inputs(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    shapes = (
        (1, 6, 2, 3),
        (1, 6, 2, 3),
        (1, 6, 4, 2),
        (1, 6, 4, 3),
        (1, 6, 4),
        (1, 4, 3, 2),
    )
    scales = (0.2, 0.18, 0.1, 0.08, 1.0, 0.07)
    inputs = [
        np.ascontiguousarray((rng.standard_normal(shape) * scale).astype(np.float32))
        for shape, scale in zip(shapes, scales, strict=True)
    ]
    inputs[4] = np.ascontiguousarray(
        rng.uniform(0.72, 1.18, shapes[4]).astype(np.float32)
    )
    return tuple(inputs)


def _independent_forward_float64(
    primals: tuple[np.ndarray, ...], *, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    query, key, prepared_u, prepared_w, gamma, initial_state = (
        item.astype(np.float64) for item in primals
    )
    batch, tokens, key_heads, key_dimension = query.shape
    value_heads, value_dimension = prepared_u.shape[2:]
    chunks = tokens // chunk_size
    heads_per_key = value_heads // key_heads
    head_map = np.arange(value_heads, dtype=np.intp) // heads_per_key
    query_chunks = np.take(
        query.reshape(batch, chunks, chunk_size, key_heads, key_dimension).transpose(
            0, 1, 3, 2, 4
        ),
        head_map,
        axis=2,
    )
    key_chunks = np.take(
        key.reshape(batch, chunks, chunk_size, key_heads, key_dimension).transpose(
            0, 1, 3, 2, 4
        ),
        head_map,
        axis=2,
    )
    u_chunks = prepared_u.reshape(
        batch, chunks, chunk_size, value_heads, value_dimension
    ).transpose(0, 1, 3, 2, 4)
    w_chunks = prepared_w.reshape(
        batch, chunks, chunk_size, value_heads, key_dimension
    ).transpose(0, 1, 3, 2, 4)
    gamma_chunks = gamma.reshape(batch, chunks, chunk_size, value_heads).transpose(
        0, 1, 3, 2
    )

    state = initial_state.copy()
    output_chunks = np.empty_like(u_chunks)
    for chunk_index in range(chunks):
        query_chunk = query_chunks[:, chunk_index]
        key_chunk = key_chunks[:, chunk_index]
        gamma_chunk = gamma_chunks[:, chunk_index]
        decay = np.tril(gamma_chunk[..., :, None] / gamma_chunk[..., None, :])
        corrected = u_chunks[:, chunk_index] - w_chunks[:, chunk_index] @ state
        attention = (query_chunk @ np.swapaxes(key_chunk, -1, -2)) * decay
        output_chunks[:, chunk_index] = (
            query_chunk * gamma_chunk[..., :, None]
        ) @ state + attention @ corrected
        reverse_key = key_chunk * decay[..., -1, :, None]
        state = (
            gamma_chunk[..., -1, None, None] * state
            + np.swapaxes(reverse_key, -1, -2) @ corrected
        )
    output = output_chunks.transpose(0, 1, 3, 2, 4).reshape(
        batch, tokens, value_heads, value_dimension
    )
    return output, state


def _objective_float64(
    primals: tuple[np.ndarray, ...],
    output_cotangent: np.ndarray,
    state_cotangent: np.ndarray,
) -> float:
    # Three rows distinguish a non-last strict-lower gamma ratio from the
    # final-row ratios that also participate in the state transition.
    output, final_state = _independent_forward_float64(primals, chunk_size=3)
    return float(
        np.sum(output * output_cotangent.astype(np.float64))
        + np.sum(final_state * state_cotangent.astype(np.float64))
    )


def test_source_is_import_light_and_pins_reverse_equations_and_residual_policy() -> (
    None
):
    source = _ORACLE_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    imported_modules = {
        node.module.partition(".")[0]
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name.partition(".")[0]
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert imported_modules.isdisjoint(
        {"jax", "jaxlib", "torch", "tilelang", "hip", "ml_dtypes", "skyrl"}
    )
    assert "S_next = gamma[-1] * S + P.T @ C" in source
    assert "hv // 2" in source
    assert "No additional internal forward residual is required" in source
    assert "27,328,512 bytes" in source and "26.0625 MiB" in source
    assert "14 MiB" in source and "16 MiB" in source
    assert "BF16 output cotangent" in source
    assert "widened internally" in source


def test_exact_geometry_byte_costs_and_named_gradient_contract_are_pinned() -> None:
    inputs = _exact_inputs()
    assert tuple(item.shape for item in inputs) == (
        (1, 512, 16, 128),
        (1, 512, 16, 128),
        (1, 512, 32, 128),
        (1, 512, 32, 128),
        (1, 512, 32),
        (1, 32, 128, 128),
        (1, 512, 32, 128),
        (1, 32, 128, 128),
    )
    assert all(item.dtype == np.float32 for item in (*inputs[:6], inputs[7]))
    assert inputs[6].dtype == reverse._bfloat16_dtype()
    assert all(item.flags.c_contiguous for item in inputs)
    assert inputs[6].nbytes == reverse.GDN_EXECUTE_S512_OUTPUT_COTANGENT_BYTES
    assert inputs[7].nbytes == reverse.GDN_EXECUTE_S512_STATE_COTANGENT_BYTES
    assert (
        inputs[6].nbytes + inputs[7].nbytes
        == reverse.GDN_EXECUTE_S512_COTANGENT_INPUT_BYTES
        == 6_291_456
    )
    assert sum(item.nbytes for item in inputs[:6]) == (
        reverse.GDN_EXECUTE_S512_PRIMAL_BYTES
    )
    assert reverse.GDN_EXECUTE_S512_PRIMAL_BYTES == 27_328_512
    assert reverse.GDN_EXECUTE_S512_GRADIENT_BYTES == 27_328_512
    assert reverse.GDN_EXECUTE_S512_INTERNAL_STATE_COUNT == 7
    assert reverse.GDN_EXECUTE_S512_INTERNAL_STATE_RESIDUAL_BYTES == (
        7 * inputs[5].nbytes
    )
    assert reverse.GDN_EXECUTE_S512_INTERNAL_STATE_RESIDUAL_BYTES == 14_680_064
    assert reverse.GDN_EXECUTE_S512_ALL_CHUNK_START_BYTES == 8 * inputs[5].nbytes
    assert reverse.GDN_EXECUTE_S512_ALL_CHUNK_START_BYTES == 16_777_216
    assert tuple(
        field.name for field in fields(reverse.GDNExecuteBoundaryGradients)
    ) == (_GRADIENT_NAMES)


def test_every_exact_input_rejects_wrong_shape_dtype_and_contiguity() -> None:
    inputs = _exact_inputs()
    names = (*_GRADIENT_NAMES, "output_cotangent", "final_state_cotangent")
    for index, name in enumerate(names):
        wrong_shape = list(inputs)
        wrong_shape[index] = inputs[index].reshape(-1)[:-1].copy()
        with pytest.raises(ValueError, match=rf"{name} shape must be exactly"):
            reverse.gdn_execute_s512_reverse_numpy(*wrong_shape)

        wrong_dtype = list(inputs)
        wrong_dtype[index] = inputs[index].astype(
            np.float32 if index == 6 else np.float64
        )
        expected_dtype = "bfloat16" if index == 6 else "float32"
        with pytest.raises(
            TypeError, match=rf"{name} dtype must be exactly {expected_dtype}"
        ):
            reverse.gdn_execute_s512_reverse_numpy(*wrong_dtype)

        noncontiguous = list(inputs)
        noncontiguous[index] = inputs[index][..., ::-1]
        with pytest.raises(ValueError, match=rf"{name} must be C-contiguous"):
            reverse.gdn_execute_s512_reverse_numpy(*noncontiguous)


def test_exact_inputs_are_numpy_disjoint_and_gamma_is_positive_finite() -> None:
    inputs = _exact_inputs()
    not_array = list(inputs)
    not_array[0] = []
    with pytest.raises(TypeError, match="query must be a NumPy array"):
        reverse.gdn_execute_s512_reverse_numpy(*not_array)

    aliased_output_bar = list(inputs)
    aliased_output_bar[6] = (
        aliased_output_bar[2]
        .view(reverse._bfloat16_dtype())
        .reshape(-1)[: np.prod(reverse.GDN_EXECUTE_S512_OUTPUT_SHAPE)]
        .reshape(reverse.GDN_EXECUTE_S512_OUTPUT_SHAPE)
    )
    with pytest.raises(ValueError, match="prepared_u overlaps output_cotangent"):
        reverse.gdn_execute_s512_reverse_numpy(*aliased_output_bar)

    aliased_state_bar = list(inputs)
    aliased_state_bar[7] = aliased_state_bar[5]
    with pytest.raises(
        ValueError, match="initial_state overlaps final_state_cotangent"
    ):
        reverse.gdn_execute_s512_reverse_numpy(*aliased_state_bar)

    for invalid in (np.float32(0.0), np.float32(-0.2), np.float32(np.inf)):
        invalid_gamma = list(inputs)
        invalid_gamma[4] = invalid_gamma[4].copy()
        invalid_gamma[4][0, 7, 3] = invalid
        with pytest.raises(ValueError, match="finite, strictly positive"):
            reverse.gdn_execute_s512_reverse_numpy(*invalid_gamma)


@pytest.mark.parametrize("cotangent_path", ["output", "final_state"])
def test_every_boundary_gradient_matches_independent_float64_finite_difference(
    cotangent_path: str,
) -> None:
    rng = np.random.default_rng(7310 + len(cotangent_path))
    primals = _reduced_inputs(5127)
    if cotangent_path == "output":
        output_cotangent = (rng.standard_normal(primals[2].shape) * 0.3).astype(
            np.float32
        )
        state_cotangent = np.zeros(primals[5].shape, dtype=np.float32)
    else:
        output_cotangent = np.zeros(primals[2].shape, dtype=np.float32)
        state_cotangent = (rng.standard_normal(primals[5].shape) * 0.3).astype(
            np.float32
        )
    gradients = reverse._gdn_execute_reverse_chunks_numpy(
        *primals,
        np.ascontiguousarray(output_cotangent),
        np.ascontiguousarray(state_cotangent),
        chunk_size=3,
    )

    epsilon = 1e-5
    for index, analytic in enumerate(_gradient_arrays(gradients)):
        direction = rng.standard_normal(primals[index].shape)
        direction /= np.linalg.norm(direction)
        plus = tuple(item.astype(np.float64) for item in primals)
        minus = tuple(item.astype(np.float64) for item in primals)
        plus = tuple(
            item + epsilon * direction if item_index == index else item
            for item_index, item in enumerate(plus)
        )
        minus = tuple(
            item - epsilon * direction if item_index == index else item
            for item_index, item in enumerate(minus)
        )
        numerical = (
            _objective_float64(plus, output_cotangent, state_cotangent)
            - _objective_float64(minus, output_cotangent, state_cotangent)
        ) / (2.0 * epsilon)
        projected_analytic = float(np.sum(analytic.astype(np.float64) * direction))
        np.testing.assert_allclose(
            projected_analytic,
            numerical,
            rtol=3e-5,
            atol=2e-7,
            err_msg=f"{cotangent_path} path for {_GRADIENT_NAMES[index]}",
        )


def test_reverse_is_linear_in_both_result_cotangents() -> None:
    rng = np.random.default_rng(8891)
    primals = _reduced_inputs(8892)
    output_bars = [
        np.ascontiguousarray(
            (rng.standard_normal(primals[2].shape) * 0.1).astype(np.float32)
        )
        for _ in range(2)
    ]
    state_bars = [
        np.ascontiguousarray(
            (rng.standard_normal(primals[5].shape) * 0.1).astype(np.float32)
        )
        for _ in range(2)
    ]
    separate = [
        reverse._gdn_execute_reverse_chunks_numpy(
            *primals, output_bars[index], state_bars[index], chunk_size=2
        )
        for index in range(2)
    ]
    combined = reverse._gdn_execute_reverse_chunks_numpy(
        *primals,
        np.ascontiguousarray(output_bars[0] + output_bars[1]),
        np.ascontiguousarray(state_bars[0] + state_bars[1]),
        chunk_size=2,
    )
    for combined_array, left, right in zip(
        _gradient_arrays(combined),
        _gradient_arrays(separate[0]),
        _gradient_arrays(separate[1]),
        strict=True,
    ):
        np.testing.assert_allclose(combined_array, left + right, rtol=2e-6, atol=2e-7)


def test_value_head_cotangents_accumulate_into_shared_key_head_hv_div_two() -> None:
    query = np.zeros((1, 2, 2, 2), dtype=np.float32)
    key = np.zeros_like(query)
    prepared_u = np.zeros((1, 2, 4, 1), dtype=np.float32)
    prepared_w = np.zeros((1, 2, 4, 2), dtype=np.float32)
    gamma = np.ones((1, 2, 4), dtype=np.float32)
    initial_state = np.zeros((1, 4, 2, 1), dtype=np.float32)
    initial_state[0, :, 0, 0] = np.asarray([1.0, 2.0, 4.0, 8.0])
    output_bar = np.zeros_like(prepared_u)
    output_bar[0, 0, :, 0] = np.asarray([3.0, 5.0, 7.0, 11.0])
    state_bar = np.zeros_like(initial_state)

    gradients = reverse._gdn_execute_reverse_chunks_numpy(
        query,
        key,
        prepared_u,
        prepared_w,
        gamma,
        initial_state,
        output_bar,
        state_bar,
        chunk_size=1,
    )
    expected = np.zeros_like(query)
    expected[0, 0, 0, 0] = 3.0 * 1.0 + 5.0 * 2.0
    expected[0, 0, 1, 0] = 7.0 * 4.0 + 11.0 * 8.0
    np.testing.assert_array_equal(gradients.query, expected)
    for item in _gradient_arrays(gradients)[1:]:
        np.testing.assert_array_equal(item, np.zeros_like(item))


@pytest.fixture(scope="module")
def exact_state_carry_reverse() -> tuple[
    tuple[np.ndarray, ...], reverse.GDNExecuteBoundaryGradients
]:
    inputs = list(_exact_inputs())
    initial_state = inputs[5]
    output_bar = inputs[6]
    state_bar = inputs[7]
    for value_head in range(32):
        initial_state[0, value_head, 0, 0] = np.float32((value_head + 1) / 100)
        state_bar[0, value_head, 0, 0] = np.float32((value_head + 1) / 50)
        for chunk_index in range(8):
            output_bar[0, chunk_index * 64, value_head, 0] = np.float32(
                (value_head + 1) * (chunk_index + 1) / 1000
            )
    exact = tuple(inputs)
    before = _hash_arrays(exact)
    gradients = reverse.gdn_execute_s512_reverse_numpy(*exact)
    assert _hash_arrays(exact) == before
    return exact, gradients


def test_full_s512_returns_fresh_exact_fp32_c_contiguous_gradients(
    exact_state_carry_reverse: tuple[
        tuple[np.ndarray, ...], reverse.GDNExecuteBoundaryGradients
    ],
) -> None:
    inputs, gradients = exact_state_carry_reverse
    arrays = _gradient_arrays(gradients)
    assert tuple(item.shape for item in arrays) == tuple(
        item.shape for item in inputs[:6]
    )
    assert all(item.dtype == np.float32 for item in arrays)
    assert all(item.flags.c_contiguous for item in arrays)
    assert sum(item.nbytes for item in arrays) == 27_328_512
    assert not any(
        np.shares_memory(gradient, item) for gradient in arrays for item in inputs
    )
    assert not any(
        np.shares_memory(arrays[left], arrays[right])
        for left in range(len(arrays))
        for right in range(left + 1, len(arrays))
    )


def test_full_s512_eight_chunk_state_carry_and_hv_div_two_sensitivity(
    exact_state_carry_reverse: tuple[
        tuple[np.ndarray, ...], reverse.GDNExecuteBoundaryGradients
    ],
) -> None:
    inputs, gradients = exact_state_carry_reverse
    initial_state, output_bar, state_bar = inputs[5:]

    expected_query = np.zeros_like(inputs[0])
    for chunk_index in range(8):
        token = chunk_index * 64
        for key_head in range(16):
            for value_head in (2 * key_head, 2 * key_head + 1):
                expected_query[0, token, key_head, 0] += (
                    output_bar[0, token, value_head, 0]
                    * initial_state[0, value_head, 0, 0]
                )
    np.testing.assert_array_equal(gradients.query, expected_query)

    expected_gamma = np.zeros_like(inputs[4])
    for chunk_index in range(8):
        expected_gamma[0, chunk_index * 64 + 63] = (
            state_bar[0, :, 0, 0] * initial_state[0, :, 0, 0]
        )
    np.testing.assert_array_equal(gradients.gamma, expected_gamma)
    np.testing.assert_array_equal(gradients.initial_state, state_bar)
    for item in (gradients.key, gradients.prepared_u, gradients.prepared_w):
        np.testing.assert_array_equal(item, np.zeros_like(item))
