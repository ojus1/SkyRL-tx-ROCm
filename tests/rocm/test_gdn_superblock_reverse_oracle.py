from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from skyrl.tx.kernels.rocm import gdn_superblock_reverse_oracle as reverse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORACLE_SOURCE = (
    _REPO_ROOT
    / "skyrl"
    / "tx"
    / "kernels"
    / "rocm"
    / "gdn_superblock_reverse_oracle.py"
)
_GRADIENT_NAMES = ("query", "key", "value", "g", "beta", "initial_state")


def _hash_arrays(arrays: tuple[np.ndarray, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(array.view(np.uint8)).hexdigest() for array in arrays)


def _gradient_arrays(
    gradients: reverse.GDNSuperblockBoundaryGradients,
) -> tuple[np.ndarray, ...]:
    return tuple(getattr(gradients, name) for name in _GRADIENT_NAMES)


def _reduced_inputs(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    shapes = (
        (1, 6, 2, 3),
        (1, 6, 2, 3),
        (1, 6, 4, 2),
        (1, 6, 4),
        (1, 6, 4),
        (1, 4, 3, 2),
    )
    scales = (0.18, 0.16, 0.08, 1.0, 1.0, 0.06)
    arrays = [
        (rng.standard_normal(shape) * scale).astype(np.float32)
        for shape, scale in zip(shapes, scales, strict=True)
    ]
    arrays[3] = -rng.uniform(0.002, 0.025, shapes[3]).astype(np.float32)
    arrays[4] = rng.uniform(0.04, 0.26, shapes[4]).astype(np.float32)
    return tuple(np.ascontiguousarray(item) for item in arrays)


def _independent_forward_float64(
    primals: tuple[np.ndarray, ...],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    query, key, value, g, beta, initial_state = (
        item.astype(np.float64) for item in primals
    )
    batch, tokens, key_heads, key_dimension = query.shape
    value_heads, value_dimension = value.shape[2:]
    chunks = tokens // chunk_size
    heads_per_key = value_heads // key_heads
    head_map = np.arange(value_heads, dtype=np.intp) // heads_per_key

    key_chunks = key.reshape(chunks, chunk_size, key_heads, key_dimension).transpose(
        0, 2, 1, 3
    )
    value_chunks = value.reshape(
        chunks,
        chunk_size,
        key_heads,
        heads_per_key,
        value_dimension,
    ).transpose(0, 2, 3, 1, 4)
    g_chunks = g.reshape(chunks, chunk_size, key_heads, heads_per_key).transpose(
        0, 2, 3, 1
    )
    beta_chunks = beta.reshape(chunks, chunk_size, key_heads, heads_per_key).transpose(
        0, 2, 3, 1
    )

    key_gram = np.einsum("chid,chjd->chij", key_chunks, key_chunks, optimize=False)
    prefix = np.cumsum(g_chunks, axis=-1)
    gamma_chunks = np.exp(prefix)
    decay = np.tril(np.exp(prefix[..., :, None] - prefix[..., None, :]))
    strict_lower = np.tril(
        beta_chunks[..., :, None] * key_gram[:, :, None, :, :] * decay,
        k=-1,
    )
    rhs = np.concatenate(
        (
            beta_chunks[..., None] * value_chunks,
            beta_chunks[..., None]
            * gamma_chunks[..., None]
            * key_chunks[:, :, None, :, :],
        ),
        axis=-1,
    )
    solution = np.linalg.solve(strict_lower + np.eye(chunk_size), rhs)
    u_chunks = solution[..., :value_dimension]
    w_chunks = solution[..., value_dimension:]

    query_chunks = np.take(
        query.reshape(batch, chunks, chunk_size, key_heads, key_dimension).transpose(
            0, 1, 3, 2, 4
        ),
        head_map,
        axis=2,
    )
    key_execute_chunks = np.take(
        key.reshape(batch, chunks, chunk_size, key_heads, key_dimension).transpose(
            0, 1, 3, 2, 4
        ),
        head_map,
        axis=2,
    )
    u_execute_chunks = (
        u_chunks.transpose(0, 3, 1, 2, 4)
        .reshape(
            batch,
            chunks,
            chunk_size,
            value_heads,
            value_dimension,
        )
        .transpose(0, 1, 3, 2, 4)
    )
    w_execute_chunks = (
        w_chunks.transpose(0, 3, 1, 2, 4)
        .reshape(
            batch,
            chunks,
            chunk_size,
            value_heads,
            key_dimension,
        )
        .transpose(0, 1, 3, 2, 4)
    )
    gamma_execute_chunks = (
        gamma_chunks.transpose(0, 3, 1, 2)
        .reshape(
            batch,
            chunks,
            chunk_size,
            value_heads,
        )
        .transpose(0, 1, 3, 2)
    )

    state = initial_state.copy()
    output_chunks = np.empty(
        (batch, chunks, value_heads, chunk_size, value_dimension),
        dtype=np.float64,
    )
    for chunk_index in range(chunks):
        query_chunk = query_chunks[:, chunk_index]
        key_chunk = key_execute_chunks[:, chunk_index]
        u_chunk = u_execute_chunks[:, chunk_index]
        w_chunk = w_execute_chunks[:, chunk_index]
        gamma_chunk = gamma_execute_chunks[:, chunk_index]
        chunk_decay = np.tril(gamma_chunk[..., :, None] / gamma_chunk[..., None, :])
        corrected = u_chunk - w_chunk @ state
        attention = (query_chunk @ np.swapaxes(key_chunk, -1, -2)) * chunk_decay
        output_chunks[:, chunk_index] = (
            query_chunk * gamma_chunk[..., :, None]
        ) @ state + attention @ corrected
        reverse_key = key_chunk * chunk_decay[..., -1, :, None]
        state = (
            gamma_chunk[..., -1, None, None] * state
            + np.swapaxes(reverse_key, -1, -2) @ corrected
        )
    output = output_chunks.transpose(0, 1, 3, 2, 4).reshape(value.shape)
    return output, state


def _objective_float64(
    primals: tuple[np.ndarray, ...],
    output_cotangent: np.ndarray,
    state_cotangent: np.ndarray,
) -> float:
    output, state = _independent_forward_float64(primals, chunk_size=3)
    return float(
        np.sum(output * output_cotangent.astype(np.float64))
        + np.sum(state * state_cotangent.astype(np.float64))
    )


@pytest.fixture(scope="module")
def exact_zero_inputs() -> tuple[np.ndarray, ...]:
    bfloat16 = reverse.execute_reverse._bfloat16_dtype()
    return (
        np.zeros(reverse.GDN_SUPERBLOCK_S512_QUERY_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_SUPERBLOCK_S512_QUERY_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_SUPERBLOCK_S512_VALUE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_SUPERBLOCK_S512_GATE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_SUPERBLOCK_S512_GATE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_SUPERBLOCK_S512_STATE_SHAPE, dtype=np.float32),
        np.zeros(reverse.GDN_SUPERBLOCK_S512_OUTPUT_SHAPE, dtype=bfloat16),
        np.zeros(reverse.GDN_SUPERBLOCK_S512_STATE_SHAPE, dtype=np.float32),
    )


def test_source_and_component_dependencies_import_no_jax_or_gpu_runtime() -> None:
    sources = (_ORACLE_SOURCE,)
    sources += tuple(
        Path(module.__file__)
        for module in (
            reverse.prepare_forward,
            reverse.execute_reverse,
            reverse.prepare_reverse,
        )
    )
    forbidden = {"jax", "jaxlib", "torch", "tilelang", "hip"}
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
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
        assert imported_roots.isdisjoint(forbidden), source_path

    source = _ORACLE_SOURCE.read_text(encoding="utf-8")
    assert "does not accept or save prepared U, W, or gamma" in source
    assert "execute dK followed by prepare" in source
    assert "Normalization and masking" in source
    assert "gdn_prepare_ffi" not in source and "gdn_execute_ffi" not in source
    signature = inspect.signature(reverse.gdn_superblock_s512_reverse_numpy)
    assert tuple(signature.parameters) == (
        "query",
        "key",
        "value",
        "g",
        "beta",
        "initial_state",
        "output_cotangent",
        "final_state_cotangent",
    )


@pytest.mark.parametrize("cotangent_path", ("output", "final_state"))
def test_every_reduced_primal_matches_independent_composed_float64_fd(
    cotangent_path: str,
) -> None:
    rng = np.random.default_rng(14500 + len(cotangent_path))
    primals = _reduced_inputs(14521)
    if cotangent_path == "output":
        output_cotangent = np.ascontiguousarray(
            (rng.standard_normal(primals[2].shape) * 0.17).astype(np.float32)
        )
        state_cotangent = np.zeros(primals[5].shape, dtype=np.float32)
    else:
        output_cotangent = np.zeros(primals[2].shape, dtype=np.float32)
        state_cotangent = np.ascontiguousarray(
            (rng.standard_normal(primals[5].shape) * 0.15).astype(np.float32)
        )
    gradients = reverse._gdn_superblock_reverse_numpy(
        *primals,
        output_cotangent,
        state_cotangent,
        chunk_size=3,
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
            _objective_float64(plus, output_cotangent, state_cotangent)
            - _objective_float64(minus, output_cotangent, state_cotangent)
        ) / (2.0 * epsilon)
        projected_analytic = float(np.sum(analytic.astype(np.float64) * direction))
        np.testing.assert_allclose(
            projected_analytic,
            numerical,
            rtol=4e-5,
            atol=3e-7,
            err_msg=f"{cotangent_path} for {_GRADIENT_NAMES[primal_index]}",
        )


def test_final_state_cotangent_reaches_first_chunk_but_not_query() -> None:
    rng = np.random.default_rng(18831)
    primals = _reduced_inputs(18832)
    output_cotangent = np.zeros(primals[2].shape, dtype=np.float32)
    state_cotangent = np.ascontiguousarray(
        (rng.standard_normal(primals[5].shape) * 0.2).astype(np.float32)
    )
    gradients = reverse._gdn_superblock_reverse_numpy(
        *primals,
        output_cotangent,
        state_cotangent,
        chunk_size=3,
    )

    np.testing.assert_array_equal(gradients.query, np.zeros_like(gradients.query))
    assert np.linalg.norm(gradients.key[:, :3]) > 0.0
    assert np.linalg.norm(gradients.value[:, :3]) > 0.0
    assert np.linalg.norm(gradients.g[:, :3]) > 0.0
    assert np.linalg.norm(gradients.beta[:, :3]) > 0.0
    assert np.linalg.norm(gradients.initial_state) > 0.0


def test_generic_zero_key_head_shape_fails_cleanly_before_divisibility() -> None:
    query = np.zeros((1, 3, 0, 2), dtype=np.float32)
    value = np.zeros((1, 3, 0, 2), dtype=np.float32)
    gate = np.zeros((1, 3, 0), dtype=np.float32)
    state = np.zeros((1, 0, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        reverse._gdn_superblock_reverse_numpy(
            query,
            query.copy(),
            value,
            gate,
            gate.copy(),
            state,
            value.copy(),
            state.copy(),
            chunk_size=3,
        )


def test_exact_s512_abi_bytes_dtypes_and_all_input_rejections(
    exact_zero_inputs: tuple[np.ndarray, ...],
) -> None:
    assert tuple(item.shape for item in exact_zero_inputs) == (
        (1, 512, 16, 128),
        (1, 512, 16, 128),
        (1, 512, 32, 128),
        (1, 512, 32),
        (1, 512, 32),
        (1, 32, 128, 128),
        (1, 512, 32, 128),
        (1, 32, 128, 128),
    )
    assert sum(item.nbytes for item in exact_zero_inputs[:6]) == (
        reverse.GDN_SUPERBLOCK_S512_PRIMAL_BYTES
    )
    assert sum(item.nbytes for item in exact_zero_inputs[6:]) == (
        reverse.GDN_SUPERBLOCK_S512_COTANGENT_BYTES
    )
    assert sum(item.nbytes for item in exact_zero_inputs) == (
        reverse.GDN_SUPERBLOCK_S512_REVERSE_INPUT_BYTES
    )
    assert (
        tuple(field.name for field in fields(reverse.GDNSuperblockBoundaryGradients))
        == _GRADIENT_NAMES
    )

    names = (
        "query",
        "key",
        "value",
        "g",
        "beta",
        "initial_state",
        "output_cotangent",
        "final_state_cotangent",
    )
    for index, name in enumerate(names):
        wrong_shape = list(exact_zero_inputs)
        wrong_shape[index] = wrong_shape[index].reshape(-1)[:-1].copy()
        with pytest.raises(ValueError, match=rf"{name} shape must be exactly"):
            reverse.gdn_superblock_s512_reverse_numpy(*wrong_shape)

        wrong_dtype = list(exact_zero_inputs)
        wrong_dtype[index] = wrong_dtype[index].astype(
            np.float32 if index == 6 else np.float64
        )
        expected_dtype = "bfloat16" if index == 6 else "float32"
        with pytest.raises(
            TypeError,
            match=rf"{name} dtype must be exactly {expected_dtype}",
        ):
            reverse.gdn_superblock_s512_reverse_numpy(*wrong_dtype)

        noncontiguous = list(exact_zero_inputs)
        noncontiguous[index] = noncontiguous[index][..., ::-1]
        with pytest.raises(ValueError, match=rf"{name} must be C-contiguous"):
            reverse.gdn_superblock_s512_reverse_numpy(*noncontiguous)

    overlapping = list(exact_zero_inputs)
    overlapping[1] = overlapping[0]
    with pytest.raises(ValueError, match="query overlaps key"):
        reverse.gdn_superblock_s512_reverse_numpy(*overlapping)


def test_exact_component_order_bf16_widening_key_sum_and_owned_outputs(
    exact_zero_inputs: tuple[np.ndarray, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [item.copy() for item in exact_zero_inputs]
    inputs[0][0, 70, 3, 4] = 11.0
    inputs[1][0, 135, 5, 6] = 12.0
    inputs[2][0, 200, 10, 7] = 13.0
    inputs[3][0, 265, 11] = 14.0
    inputs[4][0, 330, 12] = 15.0
    inputs[5][0, 13, 8, 9] = 16.0
    inputs[6][0, 395, 17, 10] = np.asarray(
        1.503,
        dtype=reverse.execute_reverse._bfloat16_dtype(),
    )
    inputs[7][0, 18, 11, 12] = 17.0
    before = _hash_arrays(tuple(inputs))
    calls: list[str] = []
    widened: list[np.float32] = []

    prepared_u = np.zeros(reverse.GDN_SUPERBLOCK_S512_VALUE_SHAPE, dtype=np.float32)
    prepared_w = np.zeros(reverse.GDN_SUPERBLOCK_S512_VALUE_SHAPE, dtype=np.float32)
    gamma = np.ones(reverse.GDN_SUPERBLOCK_S512_GATE_SHAPE, dtype=np.float32)
    prepared_u[0, 1, 2, 3] = 21.0
    prepared_w[0, 4, 5, 6] = 22.0
    gamma[0, 7, 8] = 0.75

    def fake_prepare(
        key: np.ndarray,
        value: np.ndarray,
        g: np.ndarray,
        beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append("prepare_forward")
        assert key[0, 135, 5, 6] == 12.0
        assert value[0, 200, 10, 7] == 13.0
        assert g[0, 265, 11] == 14.0
        assert beta[0, 330, 12] == 15.0
        return prepared_u, prepared_w, gamma

    execute_query_bar = np.zeros_like(inputs[0])
    execute_key_bar = np.zeros_like(inputs[1])
    execute_u_bar = np.zeros_like(prepared_u)
    execute_w_bar = np.zeros_like(prepared_w)
    execute_gamma_bar = np.zeros_like(gamma)
    execute_state_bar = np.zeros_like(inputs[5])
    execute_query_bar[0, 70, 3, 4] = 101.0
    execute_key_bar[0, 135, 5, 6] = 1.25
    execute_u_bar[0, 1, 2, 3] = 31.0
    execute_w_bar[0, 4, 5, 6] = 32.0
    execute_gamma_bar[0, 7, 8] = 33.0
    execute_state_bar[0, 18, 11, 12] = 106.0

    def fake_execute_chunks(
        query: np.ndarray,
        key: np.ndarray,
        actual_u: np.ndarray,
        actual_w: np.ndarray,
        actual_gamma: np.ndarray,
        initial_state: np.ndarray,
        output_cotangent: np.ndarray,
        final_state_cotangent: np.ndarray,
        *,
        chunk_size: int,
    ) -> reverse.execute_reverse.GDNExecuteBoundaryGradients:
        calls.append("execute_reverse")
        assert chunk_size == 64
        assert query[0, 70, 3, 4] == 11.0
        assert initial_state[0, 13, 8, 9] == 16.0
        assert actual_u[0, 1, 2, 3] == 21.0
        assert actual_w[0, 4, 5, 6] == 22.0
        assert actual_gamma[0, 7, 8] == 0.75
        assert output_cotangent.dtype == np.float32
        widened.append(output_cotangent[0, 395, 17, 10])
        assert final_state_cotangent[0, 18, 11, 12] == 17.0
        return reverse.execute_reverse.GDNExecuteBoundaryGradients(
            query=execute_query_bar,
            key=execute_key_bar,
            prepared_u=execute_u_bar,
            prepared_w=execute_w_bar,
            gamma=execute_gamma_bar,
            initial_state=execute_state_bar,
        )

    prepare_key_bar = np.zeros_like(inputs[1])
    prepare_value_bar = np.zeros_like(inputs[2])
    prepare_g_bar = np.zeros_like(inputs[3])
    prepare_beta_bar = np.zeros_like(inputs[4])
    prepare_key_bar[0, 135, 5, 6] = 2.5
    prepare_value_bar[0, 200, 10, 7] = 103.0
    prepare_g_bar[0, 265, 11] = 104.0
    prepare_beta_bar[0, 330, 12] = 105.0

    def fake_prepare_reverse(
        key: np.ndarray,
        value: np.ndarray,
        g: np.ndarray,
        beta: np.ndarray,
        u_bar: np.ndarray,
        w_bar: np.ndarray,
        gamma_bar: np.ndarray,
    ) -> reverse.prepare_reverse.GDNPrepareBoundaryGradients:
        calls.append("prepare_reverse")
        assert key[0, 135, 5, 6] == 12.0
        assert value[0, 200, 10, 7] == 13.0
        assert g[0, 265, 11] == 14.0
        assert beta[0, 330, 12] == 15.0
        assert u_bar[0, 1, 2, 3] == 31.0
        assert w_bar[0, 4, 5, 6] == 32.0
        assert gamma_bar[0, 7, 8] == 33.0
        return reverse.prepare_reverse.GDNPrepareBoundaryGradients(
            key=prepare_key_bar,
            value=prepare_value_bar,
            g=prepare_g_bar,
            beta=prepare_beta_bar,
        )

    monkeypatch.setattr(reverse.prepare_forward, "gdn_prepare_s512_numpy", fake_prepare)
    monkeypatch.setattr(
        reverse.execute_reverse,
        "_gdn_execute_reverse_chunks_numpy",
        fake_execute_chunks,
    )
    monkeypatch.setattr(
        reverse.prepare_reverse,
        "gdn_prepare_s512_reverse_numpy",
        fake_prepare_reverse,
    )
    gradients = reverse.gdn_superblock_s512_reverse_numpy(*inputs)

    assert calls == ["prepare_forward", "execute_reverse", "prepare_reverse"]
    expected_widened = inputs[6][0, 395, 17, 10].astype(np.float32)
    np.testing.assert_array_equal(widened, (expected_widened,))
    assert _hash_arrays(tuple(inputs)) == before
    assert gradients.query[0, 70, 3, 4] == 101.0
    assert gradients.key[0, 135, 5, 6] == 3.75
    assert gradients.value[0, 200, 10, 7] == 103.0
    assert gradients.g[0, 265, 11] == 104.0
    assert gradients.beta[0, 330, 12] == 105.0
    assert gradients.initial_state[0, 18, 11, 12] == 106.0

    arrays = _gradient_arrays(gradients)
    assert tuple(item.shape for item in arrays) == tuple(
        item.shape for item in inputs[:6]
    )
    assert all(item.dtype == np.float32 and item.flags.c_contiguous for item in arrays)
    assert sum(item.nbytes for item in arrays) == (
        reverse.GDN_SUPERBLOCK_S512_GRADIENT_BYTES
    )
    assert not any(
        np.shares_memory(gradient, source) for gradient in arrays for source in inputs
    )
    assert not any(
        np.shares_memory(arrays[left], arrays[right])
        for left in range(len(arrays))
        for right in range(left + 1, len(arrays))
    )
