from __future__ import annotations

import ast
import hashlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from skyrl.tx.kernels.rocm import gdn_prepare_reverse_oracle as reverse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORACLE_SOURCE = (
    _REPO_ROOT / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_prepare_reverse_oracle.py"
)
_GRADIENT_NAMES = ("key", "value", "g", "beta")


def _hash_arrays(arrays: tuple[np.ndarray, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(memoryview(array)).hexdigest() for array in arrays)


def _gradient_arrays(
    gradients: reverse.GDNPrepareBoundaryGradients,
) -> tuple[np.ndarray, ...]:
    return tuple(getattr(gradients, name) for name in _GRADIENT_NAMES)


def _reduced_inputs(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    shapes = (
        (1, 2, 3, 3),
        (1, 2, 2, 3, 2),
        (1, 2, 2, 3),
        (1, 2, 2, 3),
    )
    key = (rng.standard_normal(shapes[0]) * 0.13).astype(np.float32)
    value = (rng.standard_normal(shapes[1]) * 0.09).astype(np.float32)
    g = -rng.uniform(0.002, 0.025, shapes[2]).astype(np.float32)
    beta = rng.uniform(0.04, 0.28, shapes[3]).astype(np.float32)
    return tuple(np.ascontiguousarray(item) for item in (key, value, g, beta))


def _independent_forward_float64(
    primals: tuple[np.ndarray, ...],
    *,
    attention_mask_chunks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key, value, g, beta = (item.astype(np.float64) for item in primals)
    if attention_mask_chunks is not None:
        key *= attention_mask_chunks[:, None, :, None]
        value *= attention_mask_chunks[:, None, None, :, None]
        g *= attention_mask_chunks[:, None, None, :]
        beta *= attention_mask_chunks[:, None, None, :]
    chunk = key.shape[-2]
    value_dimension = value.shape[-1]
    key_gram = np.einsum("chid,chjd->chij", key, key, optimize=False)
    prefix = np.cumsum(g, axis=-1)
    gamma = np.exp(prefix)
    decay = np.tril(np.exp(prefix[..., :, None] - prefix[..., None, :]))
    strict_lower = np.tril(
        beta[..., :, None] * key_gram[:, :, None, :, :] * decay,
        k=-1,
    )
    rhs = np.concatenate(
        (
            beta[..., None] * value,
            beta[..., None] * gamma[..., None] * key[:, :, None, :, :],
        ),
        axis=-1,
    )
    solution = np.linalg.solve(strict_lower + np.eye(chunk), rhs)
    return solution[..., :value_dimension], solution[..., value_dimension:], gamma


def _objective_float64(
    primals: tuple[np.ndarray, ...],
    cotangents: tuple[np.ndarray, ...],
    *,
    attention_mask_chunks: np.ndarray | None = None,
) -> float:
    outputs = _independent_forward_float64(
        primals,
        attention_mask_chunks=attention_mask_chunks,
    )
    return float(
        sum(
            np.sum(output * cotangent.astype(np.float64))
            for output, cotangent in zip(outputs, cotangents, strict=True)
        )
    )


@pytest.fixture(scope="module")
def exact_zero_inputs() -> tuple[np.ndarray, ...]:
    return (
        np.zeros(reverse.GDN_PREPARE_S512_KEY_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_PREPARE_S512_VALUE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_PREPARE_S512_VALUE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_PREPARE_S512_VALUE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32),
    )


def test_source_is_import_light_and_pins_transpose_equations() -> None:
    source = _ORACLE_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    imported_roots = {
        alias.name.partition(".")[0]
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.partition(".")[0]
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert imported_roots.isdisjoint(
        {"jax", "jaxlib", "torch", "tilelang", "hip", "skyrl"}
    )
    assert "A.T R_bar = X_bar" in source
    assert "strictLower(-R_bar @ X.T)" in source
    assert "reverse cumulative sum" in source
    assert "np.swapaxes(unit_lower, -1, -2)" in source
    assert "from skyrl" not in source


def test_transpose_solve_satisfies_upper_system_not_forward_system() -> None:
    unit_lower = np.asarray(
        (
            ((1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (-0.1, 0.3, 1.0)),
            ((1.0, 0.0, 0.0), (-0.4, 1.0, 0.0), (0.2, 0.1, 1.0)),
        ),
        dtype=np.float32,
    )
    solution_cotangent = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 17
    rhs_cotangent = reverse._transpose_unit_lower_solve(
        unit_lower,
        solution_cotangent,
    )

    residual = np.swapaxes(unit_lower, -1, -2) @ rhs_cotangent - solution_cotangent
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2e-6)
    forward_solve = np.linalg.solve(unit_lower, solution_cotangent)
    assert not np.allclose(rhs_cotangent, forward_solve, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cotangent_path", ("prepared_u", "prepared_w", "gamma"))
def test_every_reduced_boundary_gradient_matches_float64_finite_difference(
    cotangent_path: str,
) -> None:
    rng = np.random.default_rng(7300 + len(cotangent_path))
    primals = _reduced_inputs(6421)
    output_shapes = (
        primals[1].shape,
        (*primals[1].shape[:-1], primals[0].shape[-1]),
        primals[2].shape,
    )
    cotangents = [np.zeros(shape, dtype=np.float32) for shape in output_shapes]
    selected = {"prepared_u": 0, "prepared_w": 1, "gamma": 2}[cotangent_path]
    cotangents[selected] = np.ascontiguousarray(
        (rng.standard_normal(output_shapes[selected]) * 0.2).astype(np.float32)
    )
    gradients = reverse._dense_prepare_reverse_chunks_numpy(
        *primals,
        *cotangents,
    )

    epsilon = 2e-5
    for primal_index, analytic in enumerate(_gradient_arrays(gradients)):
        direction = rng.standard_normal(primals[primal_index].shape)
        direction /= np.linalg.norm(direction)
        plus = tuple(
            item.astype(np.float64)
            + (epsilon * direction if index == primal_index else 0.0)
            for index, item in enumerate(primals)
        )
        minus = tuple(
            item.astype(np.float64)
            - (epsilon * direction if index == primal_index else 0.0)
            for index, item in enumerate(primals)
        )
        numerical = (
            _objective_float64(plus, tuple(cotangents))
            - _objective_float64(minus, tuple(cotangents))
        ) / (2.0 * epsilon)
        projected_analytic = float(np.sum(analytic.astype(np.float64) * direction))
        np.testing.assert_allclose(
            projected_analytic,
            numerical,
            rtol=3e-5,
            atol=2e-7,
            err_msg=f"{cotangent_path} path for {_GRADIENT_NAMES[primal_index]}",
        )


@pytest.mark.parametrize(
    "case, mask",
    (
        ("left_padding", np.asarray(((False, True, True),), dtype=np.bool_)),
        ("interior_hole", np.asarray(((True, False, True),), dtype=np.bool_)),
        ("all_masked", np.asarray(((False, False, False),), dtype=np.bool_)),
    ),
)
def test_nonzero_uw_paths_apply_exact_mask_transpose_and_match_masked_forward_fd(
    case: str,
    mask: np.ndarray,
) -> None:
    rng = np.random.default_rng(9100 + len(case))
    primals = _reduced_inputs(9107)
    cotangents = (
        np.ascontiguousarray(
            (rng.standard_normal(primals[1].shape) * 0.16).astype(np.float32)
        ),
        np.ascontiguousarray(
            (
                rng.standard_normal((*primals[1].shape[:-1], primals[0].shape[-1]))
                * 0.14
            ).astype(np.float32)
        ),
        np.zeros(primals[2].shape, dtype=np.float32),
    )
    gradients = reverse._dense_prepare_reverse_chunks_numpy(
        *primals,
        *cotangents,
        attention_mask_chunks=mask,
    )
    masks = (
        mask[:, None, :, None],
        mask[:, None, None, :, None],
        mask[:, None, None, :],
        mask[:, None, None, :],
    )
    for gradient, broadcast_mask in zip(
        _gradient_arrays(gradients),
        masks,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.where(broadcast_mask, np.float32(0.0), gradient),
            np.zeros_like(gradient),
        )

    epsilon = 2e-5
    for primal_index, (analytic, broadcast_mask) in enumerate(
        zip(_gradient_arrays(gradients), masks, strict=True)
    ):
        direction = rng.standard_normal(primals[primal_index].shape) * broadcast_mask
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0.0:
            direction = rng.standard_normal(primals[primal_index].shape)
            projected_analytic = float(np.sum(analytic.astype(np.float64) * direction))
            numerical = (
                _objective_float64(
                    tuple(
                        item.astype(np.float64)
                        + (epsilon * direction if index == primal_index else 0.0)
                        for index, item in enumerate(primals)
                    ),
                    cotangents,
                    attention_mask_chunks=mask,
                )
                - _objective_float64(
                    tuple(
                        item.astype(np.float64)
                        - (epsilon * direction if index == primal_index else 0.0)
                        for index, item in enumerate(primals)
                    ),
                    cotangents,
                    attention_mask_chunks=mask,
                )
            ) / (2.0 * epsilon)
            np.testing.assert_allclose(
                (projected_analytic, numerical),
                (0.0, 0.0),
                rtol=0.0,
                atol=1e-12,
            )
            continue

        direction /= direction_norm
        plus = tuple(
            item.astype(np.float64)
            + (epsilon * direction if index == primal_index else 0.0)
            for index, item in enumerate(primals)
        )
        minus = tuple(
            item.astype(np.float64)
            - (epsilon * direction if index == primal_index else 0.0)
            for index, item in enumerate(primals)
        )
        numerical = (
            _objective_float64(
                plus,
                cotangents,
                attention_mask_chunks=mask,
            )
            - _objective_float64(
                minus,
                cotangents,
                attention_mask_chunks=mask,
            )
        ) / (2.0 * epsilon)
        projected_analytic = float(np.sum(analytic.astype(np.float64) * direction))
        np.testing.assert_allclose(
            projected_analytic,
            numerical,
            rtol=3e-5,
            atol=2e-7,
            err_msg=f"{case} for {_GRADIENT_NAMES[primal_index]}",
        )


def test_paired_value_head_contributions_sum_into_shared_key_head_in_fixed_order() -> (
    None
):
    rng = np.random.default_rng(11221)
    key = np.ascontiguousarray(
        (rng.standard_normal((1, 1, 3, 3)) * 0.12).astype(np.float32)
    )
    value = np.ascontiguousarray(
        (rng.standard_normal((1, 1, 2, 3, 2)) * 0.08).astype(np.float32)
    )
    g = np.ascontiguousarray(-rng.uniform(0.002, 0.02, (1, 1, 2, 3)).astype(np.float32))
    beta = np.ascontiguousarray(
        rng.uniform(0.04, 0.25, (1, 1, 2, 3)).astype(np.float32)
    )
    u_bars = [np.zeros_like(value) for _ in range(2)]
    w_shape = (*value.shape[:-1], key.shape[-1])
    w_bars = [np.zeros(w_shape, dtype=np.float32) for _ in range(2)]
    for pair in range(2):
        u_bars[pair][:, :, pair] = (
            rng.standard_normal(value[:, :, pair].shape) * 0.17
        ).astype(np.float32)
        w_bars[pair][:, :, pair] = (
            rng.standard_normal(w_bars[pair][:, :, pair].shape) * 0.13
        ).astype(np.float32)
    gamma_bar = np.zeros_like(g)

    separate = [
        reverse._dense_prepare_reverse_chunks_numpy(
            key,
            value,
            g,
            beta,
            u_bars[pair],
            w_bars[pair],
            gamma_bar,
        )
        for pair in range(2)
    ]
    combined = reverse._dense_prepare_reverse_chunks_numpy(
        key,
        value,
        g,
        beta,
        np.ascontiguousarray(u_bars[0] + u_bars[1]),
        np.ascontiguousarray(w_bars[0] + w_bars[1]),
        gamma_bar,
    )
    repeated = reverse._dense_prepare_reverse_chunks_numpy(
        key,
        value,
        g,
        beta,
        np.ascontiguousarray(u_bars[0] + u_bars[1]),
        np.ascontiguousarray(w_bars[0] + w_bars[1]),
        gamma_bar,
    )
    assert np.linalg.norm(separate[0].key) > 0.0
    assert np.linalg.norm(separate[1].key) > 0.0
    np.testing.assert_allclose(
        combined.key,
        separate[0].key + separate[1].key,
        rtol=2e-6,
        atol=2e-9,
    )
    np.testing.assert_array_equal(repeated.key, combined.key)


def test_exact_s512_abi_mask_transpose_and_gamma_reverse_direction(
    exact_zero_inputs: tuple[np.ndarray, ...],
) -> None:
    inputs = list(exact_zero_inputs)
    gamma_cotangent = inputs[-1].copy()
    gamma_cotangent[0, 63, 0] = np.float32(1.0)
    gamma_cotangent[0, 127, 0] = np.float32(2.0)
    inputs[-1] = gamma_cotangent
    mask = np.ones(reverse.GDN_PREPARE_S512_MASK_SHAPE, dtype=np.bool_)
    mask[0, (0, 5, 64, 100)] = False

    before = _hash_arrays((*inputs, mask))
    gradients = reverse.gdn_prepare_s512_reverse_numpy(
        *inputs,
        attention_mask=mask,
    )
    assert _hash_arrays((*inputs, mask)) == before

    expected_g = np.zeros(reverse.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32)
    expected_g[0, :64, 0] = 1.0
    expected_g[0, 64:128, 0] = 2.0
    expected_g *= mask[..., None]
    np.testing.assert_array_equal(gradients.g, expected_g)
    np.testing.assert_array_equal(gradients.key, np.zeros_like(gradients.key))
    np.testing.assert_array_equal(gradients.value, np.zeros_like(gradients.value))
    np.testing.assert_array_equal(gradients.beta, np.zeros_like(gradients.beta))

    arrays = _gradient_arrays(gradients)
    assert tuple(item.shape for item in arrays) == tuple(
        item.shape for item in inputs[:4]
    )
    assert all(item.dtype == np.float32 and item.flags.c_contiguous for item in arrays)
    assert (
        sum(item.nbytes for item in arrays) == reverse.GDN_PREPARE_S512_GRADIENT_BYTES
    )
    assert not any(
        np.shares_memory(gradient, source)
        for gradient in arrays
        for source in (*inputs, mask)
    )
    assert not any(
        np.shares_memory(arrays[left], arrays[right])
        for left in range(len(arrays))
        for right in range(left + 1, len(arrays))
    )


def test_exact_s512_byte_contract_names_and_validation_are_pinned(
    exact_zero_inputs: tuple[np.ndarray, ...],
) -> None:
    assert tuple(item.shape for item in exact_zero_inputs) == (
        (1, 512, 16, 128),
        (1, 512, 32, 128),
        (1, 512, 32),
        (1, 512, 32),
        (1, 512, 32, 128),
        (1, 512, 32, 128),
        (1, 512, 32),
    )
    assert sum(item.nbytes for item in exact_zero_inputs[:4]) == (
        reverse.GDN_PREPARE_S512_PRIMAL_BYTES
    )
    assert sum(item.nbytes for item in exact_zero_inputs[4:]) == (
        reverse.GDN_PREPARE_S512_COTANGENT_BYTES
    )
    assert sum(item.nbytes for item in exact_zero_inputs) == (
        reverse.GDN_PREPARE_S512_REVERSE_INPUT_BYTES
    )
    assert (
        tuple(field.name for field in fields(reverse.GDNPrepareBoundaryGradients))
        == _GRADIENT_NAMES
    )

    names = (
        "key",
        "value",
        "g",
        "beta",
        "prepared_u_cotangent",
        "prepared_w_cotangent",
        "gamma_cotangent",
    )
    for index, name in enumerate(names):
        wrong_shape = list(exact_zero_inputs)
        wrong_shape[index] = wrong_shape[index].reshape(-1)[:-1].copy()
        with pytest.raises(ValueError, match=rf"{name} shape must be exactly"):
            reverse.gdn_prepare_s512_reverse_numpy(*wrong_shape)

        wrong_dtype = list(exact_zero_inputs)
        wrong_dtype[index] = wrong_dtype[index].astype(np.float64)
        with pytest.raises(TypeError, match=rf"{name} dtype must be exactly float32"):
            reverse.gdn_prepare_s512_reverse_numpy(*wrong_dtype)

        noncontiguous = list(exact_zero_inputs)
        noncontiguous[index] = noncontiguous[index][..., ::-1]
        with pytest.raises(ValueError, match=rf"{name} must be C-contiguous"):
            reverse.gdn_prepare_s512_reverse_numpy(*noncontiguous)

    overlapping = list(exact_zero_inputs)
    overlapping[5] = overlapping[4]
    with pytest.raises(
        ValueError,
        match="prepared_u_cotangent overlaps prepared_w_cotangent",
    ):
        reverse.gdn_prepare_s512_reverse_numpy(*overlapping)


def test_exact_public_boundary_maps_token_head_pair_axes_without_dense_solve(
    exact_zero_inputs: tuple[np.ndarray, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [item.copy() for item in exact_zero_inputs]
    inputs[0][0, 70, 3, 4] = 11.0
    inputs[1][0, 135, 9, 5] = 12.0
    inputs[2][0, 200, 10] = 13.0
    inputs[3][0, 265, 13] = 14.0
    inputs[4][0, 330, 14, 6] = 15.0
    inputs[5][0, 395, 17, 7] = 16.0
    inputs[6][0, 460, 18] = 17.0
    mask = np.ones(reverse.GDN_PREPARE_S512_MASK_SHAPE, dtype=np.bool_)
    mask[0, (1, 65, 129, 193, 257, 321, 385, 449)] = False

    def fake_reverse(
        key: np.ndarray,
        value: np.ndarray,
        g: np.ndarray,
        beta: np.ndarray,
        u_bar: np.ndarray,
        w_bar: np.ndarray,
        gamma_bar: np.ndarray,
        *,
        attention_mask_chunks: np.ndarray | None,
    ) -> reverse.GDNPrepareBoundaryGradients:
        assert key[1, 3, 6, 4] == 11.0
        assert value[2, 4, 1, 7, 5] == 12.0
        assert g[3, 5, 0, 8] == 13.0
        assert beta[4, 6, 1, 9] == 14.0
        assert u_bar[5, 7, 0, 10, 6] == 15.0
        assert w_bar[6, 8, 1, 11, 7] == 16.0
        assert gamma_bar[7, 9, 0, 12] == 17.0
        np.testing.assert_array_equal(
            attention_mask_chunks,
            mask.reshape(8, 64),
        )

        key_gradient = np.zeros_like(key)
        value_gradient = np.zeros_like(value)
        g_gradient = np.zeros_like(g)
        beta_gradient = np.zeros_like(beta)
        key_gradient[3, 5, 7, 11] = 101.0
        value_gradient[4, 6, 1, 8, 12] = 202.0
        g_gradient[5, 7, 0, 9] = 303.0
        beta_gradient[6, 8, 1, 10] = 404.0
        return reverse.GDNPrepareBoundaryGradients(
            key=key_gradient,
            value=value_gradient,
            g=g_gradient,
            beta=beta_gradient,
        )

    monkeypatch.setattr(reverse, "_dense_prepare_reverse_chunks_numpy", fake_reverse)
    gradients = reverse.gdn_prepare_s512_reverse_numpy(
        *inputs,
        attention_mask=mask,
    )

    assert gradients.key[0, 199, 5, 11] == 101.0
    assert gradients.value[0, 264, 13, 12] == 202.0
    assert gradients.g[0, 329, 14] == 303.0
    assert gradients.beta[0, 394, 17] == 404.0
    assert np.count_nonzero(gradients.key) == 1
    assert np.count_nonzero(gradients.value) == 1
    assert np.count_nonzero(gradients.g) == 1
    assert np.count_nonzero(gradients.beta) == 1
