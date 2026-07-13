from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from skyrl.tx.kernels.rocm import gdn_execute_oracle as execute
from skyrl.tx.kernels.rocm import gdn_forward_oracle as forward
from skyrl.tx.kernels.rocm import gdn_prepare_oracle as prepare

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORACLE_SOURCE = (
    _REPO_ROOT / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_execute_oracle.py"
)


def _raw_inputs(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    query = (rng.standard_normal(execute.GDN_EXECUTE_S512_QUERY_SHAPE) * 0.12).astype(
        np.float32
    )
    key = (rng.standard_normal(execute.GDN_EXECUTE_S512_QUERY_SHAPE) * 0.12).astype(
        np.float32
    )
    value = (
        rng.standard_normal(execute.GDN_EXECUTE_S512_PREPARED_SHAPE) * 0.025
    ).astype(np.float32)
    g = -rng.uniform(0.0005, 0.006, execute.GDN_EXECUTE_S512_GAMMA_SHAPE).astype(
        np.float32
    )
    beta = rng.uniform(0.005, 0.07, execute.GDN_EXECUTE_S512_GAMMA_SHAPE).astype(
        np.float32
    )
    initial_state = (
        rng.standard_normal(execute.GDN_EXECUTE_S512_STATE_SHAPE) * 0.003
    ).astype(np.float32)
    return query, key, value, g, beta, initial_state


def _model_side_transform(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Mirror only the pre-stage masking/normalization boundary."""
    mask_f32 = mask.astype(np.float32, copy=False)
    query = np.asarray(query * mask_f32[..., None, None], dtype=np.float32)
    key = np.asarray(key * mask_f32[..., None, None], dtype=np.float32)
    value = np.asarray(value * mask_f32[..., None, None], dtype=np.float32)
    g = np.asarray(g * mask_f32[..., None], dtype=np.float32)
    beta = np.asarray(beta * mask_f32[..., None], dtype=np.float32)
    query /= np.sqrt(
        np.sum(query * query, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(1e-6)
    )
    key /= np.sqrt(
        np.sum(key * key, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6)
    )
    query *= np.float32(1.0 / math.sqrt(128))
    return tuple(np.ascontiguousarray(item) for item in (query, key, value, g, beta))


def _compose(
    raw: tuple[np.ndarray, ...], mask: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple]:
    query, key, value, g, beta, initial_state = raw
    transformed = _model_side_transform(query, key, value, g, beta, mask)
    (
        transformed_query,
        transformed_key,
        transformed_value,
        transformed_g,
        transformed_beta,
    ) = transformed
    prepared_u, prepared_w, gamma = prepare.gdn_prepare_s512_numpy(
        transformed_key,
        transformed_value,
        transformed_g,
        transformed_beta,
    )
    boundary = (
        transformed_query,
        transformed_key,
        prepared_u,
        prepared_w,
        gamma,
        initial_state,
    )
    before = _hash_arrays(boundary)
    actual = execute.gdn_execute_s512_numpy(
        *boundary,
        output_bfloat16=False,
    )
    assert _hash_arrays(boundary) == before
    expected = forward.recurrent_gdn_forward_numpy(
        query,
        key,
        value,
        g,
        beta,
        attention_mask=mask,
        initial_state=initial_state,
    )
    return actual, expected, boundary


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = actual.astype(np.float32) - expected.astype(np.float32)
    numerator = np.linalg.norm(difference.reshape(-1).astype(np.float64))
    denominator = np.linalg.norm(expected.reshape(-1).astype(np.float64))
    return float(numerator / max(denominator, np.finfo(np.float64).tiny))


def _cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    left = actual.reshape(-1).astype(np.float64)
    right = expected.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator)


def _hash_arrays(arrays: tuple[np.ndarray, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(memoryview(array)).hexdigest() for array in arrays)


def _zero_boundary_inputs() -> tuple[np.ndarray, ...]:
    return (
        np.zeros(execute.GDN_EXECUTE_S512_QUERY_SHAPE, dtype=np.float32),
        np.zeros(execute.GDN_EXECUTE_S512_QUERY_SHAPE, dtype=np.float32),
        np.zeros(execute.GDN_EXECUTE_S512_PREPARED_SHAPE, dtype=np.float32),
        np.zeros(execute.GDN_EXECUTE_S512_PREPARED_SHAPE, dtype=np.float32),
        np.ones(execute.GDN_EXECUTE_S512_GAMMA_SHAPE, dtype=np.float32),
        np.zeros(execute.GDN_EXECUTE_S512_STATE_SHAPE, dtype=np.float32),
    )


@pytest.fixture(scope="module")
def dense_all_valid() -> tuple:
    raw = _raw_inputs(3512)
    mask = np.ones((1, 512), dtype=np.bool_)
    return (*_compose(raw, mask), raw)


def test_source_is_import_light_and_documents_token_major_prepare_layout() -> None:
    source = _ORACLE_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0] for node in imports for alias in node.names
    }

    assert imported_roots.isdisjoint(
        {"jax", "jaxlib", "torch", "tilelang", "hip", "ml_dtypes", "skyrl"}
    )
    assert "canonical token-major layouts" in source
    assert "hv // 2" in source
    assert "gdn_prepare_s512_numpy" in source
    assert "from skyrl" not in source
    assert "linalg.solve" not in source


def test_exact_byte_constants_and_canonical_layout_are_pinned() -> None:
    inputs = _zero_boundary_inputs()
    assert tuple(array.shape for array in inputs) == (
        (1, 512, 16, 128),
        (1, 512, 16, 128),
        (1, 512, 32, 128),
        (1, 512, 32, 128),
        (1, 512, 32),
        (1, 32, 128, 128),
    )
    assert all(array.dtype == np.float32 for array in inputs)
    assert all(array.flags.c_contiguous for array in inputs)
    assert sum(array.nbytes for array in inputs) == 27_328_512
    assert execute.GDN_EXECUTE_S512_INPUT_BYTES == 27_328_512
    assert execute.GDN_EXECUTE_S512_BF16_OUTPUT_BYTES == 4_194_304
    assert execute.GDN_EXECUTE_S512_STATE_BYTES == 2_097_152
    assert execute.GDN_EXECUTE_S512_OUTPUT_TENSOR_BYTES == 6_291_456
    assert execute.GDN_EXECUTE_S512_COMPILED_OUTPUT_BYTES == 6_291_472


def test_every_input_rejects_wrong_shape_dtype_and_contiguity() -> None:
    inputs = _zero_boundary_inputs()
    names = ("query", "key", "prepared_u", "prepared_w", "gamma", "initial_state")

    for index, name in enumerate(names):
        wrong_shape = list(inputs)
        wrong_shape[index] = inputs[index].reshape(-1)[:-1].copy()
        with pytest.raises(ValueError, match=rf"{name} shape must be exactly"):
            execute.gdn_execute_s512_numpy(*wrong_shape)

        wrong_dtype = list(inputs)
        wrong_dtype[index] = inputs[index].astype(np.float64)
        with pytest.raises(TypeError, match=rf"{name} dtype must be exactly float32"):
            execute.gdn_execute_s512_numpy(*wrong_dtype)

        noncontiguous = list(inputs)
        noncontiguous[index] = inputs[index][..., ::-1]
        with pytest.raises(ValueError, match=rf"{name} must be C-contiguous"):
            execute.gdn_execute_s512_numpy(*noncontiguous)

    chunk_major = list(inputs)
    chunk_major[2] = chunk_major[2].reshape(8, 64, 32, 128)
    with pytest.raises(ValueError, match="prepared_u shape must be exactly"):
        execute.gdn_execute_s512_numpy(*chunk_major)


def test_inputs_must_be_numpy_arrays_and_pairwise_non_overlapping() -> None:
    inputs = _zero_boundary_inputs()
    not_array = list(inputs)
    not_array[0] = []
    with pytest.raises(TypeError, match="query must be a NumPy array"):
        execute.gdn_execute_s512_numpy(*not_array)

    aliased_qk = list(inputs)
    aliased_qk[1] = aliased_qk[0]
    with pytest.raises(ValueError, match="query overlaps key"):
        execute.gdn_execute_s512_numpy(*aliased_qk)

    aliased_uw = list(inputs)
    aliased_uw[3] = aliased_uw[2]
    with pytest.raises(ValueError, match="prepared_u overlaps prepared_w"):
        execute.gdn_execute_s512_numpy(*aliased_uw)

    cross_shape_overlap = list(inputs)
    cross_shape_overlap[4] = (
        cross_shape_overlap[0]
        .reshape(-1)[: np.prod(execute.GDN_EXECUTE_S512_GAMMA_SHAPE)]
        .reshape(execute.GDN_EXECUTE_S512_GAMMA_SHAPE)
    )
    with pytest.raises(ValueError, match="query overlaps gamma"):
        execute.gdn_execute_s512_numpy(*cross_shape_overlap)

    with pytest.raises(TypeError, match="output_bfloat16 must be an exact bool"):
        execute.gdn_execute_s512_numpy(*inputs, output_bfloat16=1)


def test_bfloat16_dependency_is_lazy_and_only_required_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _zero_boundary_inputs()

    def unavailable(name: str) -> None:
        assert name == "ml_dtypes"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(execute.importlib, "import_module", unavailable)
    output, state = execute.gdn_execute_s512_numpy(
        *inputs,
        output_bfloat16=False,
    )
    assert output.dtype == state.dtype == np.float32
    with pytest.raises(RuntimeError, match="ml_dtypes is not installed"):
        execute.gdn_execute_s512_numpy(*inputs)


def test_dense_all_valid_nonzero_state_matches_independent_recurrence_fp32(
    dense_all_valid: tuple,
) -> None:
    actual, expected, _, _ = dense_all_valid
    assert actual[0].shape == execute.GDN_EXECUTE_S512_OUTPUT_SHAPE
    assert actual[1].shape == execute.GDN_EXECUTE_S512_STATE_SHAPE
    assert actual[0].dtype == actual[1].dtype == np.float32
    assert actual[0].flags.c_contiguous and actual[1].flags.c_contiguous
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=2e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-5, atol=2e-7)


def test_bfloat16_output_boundary_and_fp32_state_meet_numerical_gates(
    dense_all_valid: tuple,
) -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    fp32_actual, _, boundary, _ = dense_all_valid
    output, state = execute.gdn_execute_s512_numpy(*boundary)

    assert output.dtype == np.dtype(ml_dtypes.bfloat16)
    assert state.dtype == np.float32
    assert output.nbytes == execute.GDN_EXECUTE_S512_BF16_OUTPUT_BYTES
    assert state.nbytes == execute.GDN_EXECUTE_S512_STATE_BYTES
    assert output.nbytes + state.nbytes == execute.GDN_EXECUTE_S512_OUTPUT_TENSOR_BYTES
    assert output.flags.c_contiguous and state.flags.c_contiguous
    assert _relative_l2(output, fp32_actual[0]) <= 5e-3
    assert _cosine(output, fp32_actual[0]) >= 0.9999
    assert np.max(np.abs(output.astype(np.float32) - fp32_actual[0])) <= 5e-3
    np.testing.assert_array_equal(state, fp32_actual[1])


def test_value_heads_map_to_paired_key_heads() -> None:
    query, key, prepared_u, prepared_w, gamma, initial_state = _zero_boundary_inputs()
    for key_head in range(16):
        query[0, 0, key_head, 0] = np.float32(key_head + 1)
    for value_head in range(32):
        initial_state[0, value_head, 0, 0] = np.float32(100 + value_head)

    output, final_state = execute.gdn_execute_s512_numpy(
        query,
        key,
        prepared_u,
        prepared_w,
        gamma,
        initial_state,
        output_bfloat16=False,
    )
    expected = np.asarray(
        [((value_head // 2) + 1) * (100 + value_head) for value_head in range(32)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(output[0, 0, :, 0], expected)
    np.testing.assert_array_equal(output[0, 1:], np.zeros_like(output[0, 1:]))
    np.testing.assert_array_equal(final_state, initial_state)


def _mask(case: str) -> np.ndarray:
    mask = np.ones((1, 512), dtype=np.bool_)
    if case.startswith("tail"):
        valid = int(case.removeprefix("tail"))
        mask[:, valid:] = False
    elif case == "interior_holes":
        mask[:, [1, 62, 63, 64, 65, 255, 256, 400]] = False
    elif case == "left_padding":
        mask[:, :65] = False
    elif case == "all_masked":
        mask[:] = False
    else:
        raise AssertionError(f"unknown mask case: {case}")
    return mask


@pytest.mark.parametrize(
    "case",
    [
        "tail63",
        "tail64",
        "tail65",
        "tail511",
        "interior_holes",
        "left_padding",
        "all_masked",
    ],
)
def test_masked_model_inputs_compose_through_prepare_and_execute(case: str) -> None:
    raw = _raw_inputs(8100 + sum(case.encode("ascii")))
    mask = _mask(case)
    actual, expected, boundary = _compose(raw, mask)

    transformed_query, transformed_key = boundary[:2]
    transformed_value, transformed_g, transformed_beta = _model_side_transform(
        *raw[:5], mask
    )[2:]
    for array in (transformed_query, transformed_key, transformed_value):
        np.testing.assert_array_equal(array[:, ~mask[0]], 0.0)
    for array in (transformed_g, transformed_beta):
        np.testing.assert_array_equal(array[:, ~mask[0]], 0.0)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=2e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-5, atol=2e-7)
    np.testing.assert_array_equal(actual[0][:, ~mask[0]], 0.0)
    assert actual[1].dtype == np.float32
    if case == "all_masked":
        np.testing.assert_array_equal(actual[1], raw[-1])


def test_repeated_execution_is_deterministic_and_does_not_mutate_inputs(
    dense_all_valid: tuple,
) -> None:
    first, _, boundary, _ = dense_all_valid
    before = _hash_arrays(boundary)
    second = execute.gdn_execute_s512_numpy(
        *boundary,
        output_bfloat16=False,
    )
    after = _hash_arrays(boundary)

    assert after == before
    np.testing.assert_array_equal(second[0], first[0])
    np.testing.assert_array_equal(second[1], first[1])
    assert not any(
        np.shares_memory(result, item) for result in second for item in boundary
    )
