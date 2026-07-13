from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import inspect
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ml_dtypes
import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_vjp.py"
_SPEC = importlib.util.spec_from_file_location("vjp_probe_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
_TARGET = "__gpu$xla.gpu.triton"


def _records(output: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _accelerator_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib"} or name.startswith(("jax.", "jaxlib."))
    }


@pytest.fixture(scope="module")
def exact_case():
    return _PROBE._construct_host_case(np, ml_dtypes)


def test_default_is_abstract_refusal_without_accelerator_import():
    before = _accelerator_modules()
    output = io.StringIO()
    assert (
        _PROBE._execute(SimpleNamespace(platform="abstract", allow_gpu=False), output)
        == 0
    )
    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["contract"] == {
        "operation": "query_bounded_gqa_t512_full_vjp_refusal",
        "exact_logical_internal_calls": ["forward_q0", "dq_q0", "dkdv_q0"],
        "gpu_work_authorized": False,
    }
    assert refused["jax_imported"] is False
    assert refused["counters"] == _PROBE._zero_counters()
    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (("--platform", "rocm", "--allow-gpu"), "requires --output"),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--sequence-length", "1024"), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--padding",), "unrecognized arguments"),
        (("--second-vjp",), "unrecognized arguments"),
        (("--compile-diagnostic",), "only valid with --platform rocm"),
        (
            ("--platform", "rocm", "--compile-diagnostic"),
            "requires the explicit --allow-gpu",
        ),
    ],
)
def test_parser_refuses_implicit_gpu_and_scope_broadening(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_exclusive_mode_0600(tmp_path):
    path = tmp_path / "vjp.jsonl"
    assert _PROBE.main(["--output", str(path)]) == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(path)])


def test_compile_diagnostic_requires_full_rocm_acknowledgement_and_private_output(
    tmp_path,
):
    path = tmp_path / "compile-diagnostic.jsonl"
    args = _PROBE._parse_args(
        [
            "--platform",
            "rocm",
            "--allow-gpu",
            "--compile-diagnostic",
            "--output",
            str(path),
        ]
    )
    assert args.platform == "rocm"
    assert args.allow_gpu is True
    assert args.compile_diagnostic is True
    assert args.output == path
    contract = _PROBE._compile_diagnostic_contract()
    assert contract["checked_capability_creation_or_release_authorized"] is False
    assert contract["host_reference_construction_authorized"] is False
    assert contract["executable_invocation_authorized"] is False
    assert contract["always_stop_after_compile_and_postflight"] is True


def test_exact_contract_fixes_shape_dispatch_memory_and_numerical_gates():
    contract = _PROBE._exact_contract()
    assert contract["operation"] == "query_bounded_gqa_t512_forward_and_full_vjp"
    assert contract["gpu_architecture"] == "gfx1100"
    assert contract["gpu_pci_device_id"] == "0x744c"
    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 512, 16, 256],
        [1, 512, 4, 256],
        [1, 512, 4, 256],
        [1, 512],
        [1, 512, 16, 256],
    ]
    assert [item["shape"] for item in contract["outputs"]] == [
        [1, 512, 16, 256],
        [1, 512, 16, 256],
        [1, 512, 4, 256],
        [1, 512, 4, 256],
    ]
    assert contract["scale_exact_fraction"] == "3/32"
    assert contract["tiles"] == {
        "query_chunk_size": 512,
        "block_q": 64,
        "block_k": 64,
        "backward_block_q": 32,
        "backward_block_k": 32,
    }
    assert contract["compile_gate"]["exact_custom_calls"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
    }
    assert contract["compile_gate"]["exact_total_custom_calls"] == 3
    assert contract["compile_gate"]["sole_target_is_exact_rocm_triton"] is True
    assert (
        contract["compile_gate"]["sole_target_sha256"]
        == hashlib.sha256(_TARGET.encode()).hexdigest()
    )
    assert (
        contract["compile_gate"]["all_calls_directly_owned_by_sole_entry_computation"]
        is True
    )
    assert contract["compile_gate"]["exact_argument_bytes"] == 10_487_808
    assert contract["compile_gate"]["exact_output_bytes"] == 10_485_792
    assert contract["compile_gate"]["exact_alias_bytes"] == 0
    execution = contract["execution_contract"]
    assert execution["logical_internal_dispatches"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
        "total": 3,
    }
    assert execution["checked_executable_invocations"] == 1
    assert all(
        execution[name] == 0
        for name in (
            "warmup_invocations",
            "replay_invocations",
            "second_vjp_invocations",
            "gpu_reference_invocations",
            "device_error_reduction_invocations",
            "model_invocations",
        )
    )


def test_static_source_binding_and_kernel_api_are_exact(monkeypatch):
    proof = _PROBE._assert_static_source_bindings()
    assert proof["passed"] is True
    assert (
        proof["delegated_compile_probe_source_sha256"]
        == "bf01187101d20362072c96f70fbb80b2f8eed88fa55e60cbf479abba6db012a2"
    )
    assert (
        proof["delegated_nonzero_probe_source_sha256"]
        == "999e027d4cc35a8d59cc294020f8865036f8fb817a847ac38f96e36b597f74ac"
    )
    path = _PROBE._source_files()["delegated_compile_probe_source_sha256"].resolve()
    original = _PROBE._file_sha256
    monkeypatch.setattr(
        _PROBE,
        "_file_sha256",
        lambda value: "0" * 64 if Path(value).resolve() == path else original(value),
    )
    with pytest.raises(RuntimeError, match="pin changed"):
        _PROBE._assert_static_source_bindings()


def test_gfx1100_drm_binding_requires_the_sole_exact_amd_pci_device(tmp_path):
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "device").write_text("0x744c\n")
    proof = _PROBE._assert_gfx1100_drm(tmp_path)
    assert proof == {
        "passed": True,
        "architecture": "gfx1100",
        "amd_pci_vendor_id": "0x1002",
        "amd_pci_device_id": "0x744c",
        "sole_amd_drm_card": "card1",
    }
    (device / "device").write_text("0x0000\n")
    with pytest.raises(RuntimeError, match="gfx1100"):
        _PROBE._assert_gfx1100_drm(tmp_path)


def test_exact_host_inputs_cotangent_hashes_norms_and_scratch(exact_case):
    inputs, manifests, expected, reference = exact_case
    q, k, v, key_mask, dout = inputs
    assert [item["name"] for item in manifests] == ["q", "k", "v", "key_mask", "dout"]
    assert {
        item["name"]: item["sha256"] for item in manifests
    } == _PROBE._EXPECTED_INPUT_SHA256
    assert {
        item["name"]: item["sha256"] for item in reference["outputs"]
    } == _PROBE._EXPECTED_REFERENCE_SHA256
    assert reference["reference_l2_norms"] == pytest.approx(
        _PROBE._EXPECTED_REFERENCE_NORMS, abs=1e-12
    )
    assert all(reference["calibration_pin_checks"].values())
    assert (
        reference["oracle"]["conservative_accounted_numpy_array_scratch_bytes"]
        == 323_072
    )
    assert reference["oracle"]["full_t_by_t_matrix_constructed"] is False
    assert reference["oracle"][
        "observed_maximum_absolute_valid_logit"
    ] == pytest.approx(1.4235591888427734, abs=1e-12)
    assert q.dtype == k.dtype == v.dtype == dout.dtype == ml_dtypes.bfloat16
    assert np.all(key_mask == 1)
    assert np.count_nonzero(dout) == dout.size
    assert not np.array_equal(q, dout)
    assert all(array.dtype == np.float32 for array in expected)


@pytest.mark.parametrize(
    ("sequence", "query_heads", "kv_heads", "head_dim", "query_tile", "key_tile"),
    ((9, 4, 2, 8, 3, 4), (17, 6, 3, 7, 5, 6), (32, 8, 2, 5, 8, 7)),
)
def test_tiled_forward_vjp_matches_independent_dense_small_reference(
    sequence, query_heads, kv_heads, head_dim, query_tile, key_tile
):
    rng = np.random.Generator(np.random.PCG64(9800 + sequence))
    q = rng.uniform(-0.75, 0.75, (1, sequence, query_heads, head_dim)).astype(
        np.float32
    )
    k = rng.uniform(-0.75, 0.75, (1, sequence, kv_heads, head_dim)).astype(np.float32)
    v = rng.uniform(-0.75, 0.75, k.shape).astype(np.float32)
    dout = rng.uniform(-0.4, 0.4, q.shape).astype(np.float32)
    mask = np.ones((1, sequence), dtype=np.int32)
    tiled, metadata = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        mask,
        dout,
        scale=3 / 32,
        query_tile=query_tile,
        key_tile=key_tile,
    )
    dense = _PROBE._dense_causal_gqa_forward_vjp_reference(
        np, q, k, v, mask, dout, scale=3 / 32
    )
    for actual, expected in zip(tiled, dense, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=6e-7)
    assert metadata["full_t_by_t_matrix_constructed"] is False
    assert metadata["accelerator_used"] is False


def test_tiled_vjp_matches_independent_float64_directional_finite_differences():
    rng = np.random.Generator(np.random.PCG64(4421))
    q = rng.uniform(-0.5, 0.5, (1, 5, 4, 3)).astype(np.float32)
    k = rng.uniform(-0.5, 0.5, (1, 5, 2, 3)).astype(np.float32)
    v = rng.uniform(-0.5, 0.5, (1, 5, 2, 3)).astype(np.float32)
    dout = rng.uniform(-0.3, 0.3, q.shape).astype(np.float32)
    mask = np.ones((1, 5), np.int32)
    (_output, dq, dk, dv), _metadata = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, dout, scale=3 / 32, query_tile=3, key_tile=2
    )

    def scalar_loss(q_item, k_item, v_item):
        q64 = np.asarray(q_item[0], np.float64)
        k64 = np.repeat(np.asarray(k_item[0], np.float64), 2, axis=1)
        v64 = np.repeat(np.asarray(v_item[0], np.float64), 2, axis=1)
        logits = np.einsum("thd,shd->hts", q64, k64) * (3 / 32)
        logits[:, np.triu(np.ones((5, 5), dtype=np.bool_), k=1)] = -np.inf
        logits -= np.max(logits, axis=2, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=2, keepdims=True)
        value = np.einsum("hts,shd->thd", probability, v64)
        return float(np.sum(value * np.asarray(dout[0], np.float64)))

    epsilon = 1e-4
    for index, (primal, gradient) in enumerate(zip((q, k, v), (dq, dk, dv))):
        direction = rng.normal(size=primal.shape).astype(np.float32)
        plus = [q, k, v]
        minus = [q, k, v]
        plus[index] = primal + epsilon * direction
        minus[index] = primal - epsilon * direction
        finite_difference = (scalar_loss(*plus) - scalar_loss(*minus)) / (2 * epsilon)
        analytic = float(
            np.sum(np.asarray(gradient, np.float64) * np.asarray(direction, np.float64))
        )
        assert analytic == pytest.approx(finite_difference, rel=2e-3, abs=2e-4)


def test_exact_t512_tiled_oracle_matches_dense_reference(exact_case):
    inputs, _manifests, expected, _reference = exact_case
    dense = _PROBE._dense_causal_gqa_forward_vjp_reference(np, *inputs, scale=3 / 32)
    for actual, dense_item in zip(expected, dense, strict=True):
        np.testing.assert_allclose(actual, dense_item, rtol=3e-6, atol=6e-7)


def test_oracle_rejects_padding_and_degenerate_shapes():
    q = np.ones((1, 8, 4, 3), np.float32)
    k = np.ones((1, 8, 2, 3), np.float32)
    v = np.ones_like(k)
    dout = np.ones_like(q)
    mask = np.ones((1, 8), np.int32)
    mask[:, -1] = 0
    with pytest.raises(RuntimeError, match="all-valid"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, mask, dout, scale=3 / 32
        )
    with pytest.raises(RuntimeError, match="int32-ones"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, np.full_like(mask, 2), dout, scale=3 / 32
        )
    with pytest.raises(RuntimeError, match="int32-ones"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, np.ones_like(mask, dtype=np.bool_), dout, scale=3 / 32
        )
    with pytest.raises(RuntimeError, match="shapes"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np,
            q,
            k[:, :, :, :2],
            v[:, :, :, :2],
            np.ones_like(mask),
            dout,
            scale=3 / 32,
        )


def _stablehlo_calls(
    *,
    target: str = _TARGET,
    metadata: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    extra: str = "",
) -> str:
    metadata = metadata or dict.fromkeys(
        _PROBE._EXPECTED_MARKERS, "query_start=0 query_size=512"
    )
    markers = markers or _PROBE._EXPECTED_MARKERS
    definitions = [
        f'#loc{index} = loc("{markers[kind]} {metadata[kind]}")'
        for index, kind in enumerate(_PROBE._EXPECTED_MARKERS)
    ]
    calls = [
        f'    %{index} = stablehlo.custom_call @"{target}"() : () -> tensor<1xbf16> loc(#loc{index})'
        for index in range(3)
    ]
    return "\n".join(
        [
            *definitions,
            "module {",
            "  func.func public @main() attributes {probe = true} {",
            *calls,
            "    return",
            "  }",
            "}",
            extra,
        ]
    )


def _optimized_hlo_calls(
    *,
    target: str = _TARGET,
    metadata: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    extra: str = "",
) -> str:
    metadata = metadata or dict.fromkeys(
        _PROBE._EXPECTED_MARKERS, "query_start=0 query_size=512"
    )
    markers = markers or _PROBE._EXPECTED_MARKERS
    calls = [
        (
            f'  %{index} = bf16[1] custom-call(), custom_call_target="{target}", '
            f'op_name="{markers[kind]} {metadata[kind]}"'
        )
        for index, kind in enumerate(_PROBE._EXPECTED_MARKERS)
    ]
    return "\n".join(
        ["ENTRY main (p0: bf16[1]{0}) -> bf16[1]{0} {", *calls, "}", extra]
    )


def _helper_owned_calls(dialect: str, *, main_invocations: int) -> str:
    if dialect == "stablehlo":
        definitions = [
            f'#loc{index} = loc("{marker} query_start=0 query_size=512")'
            for index, marker in enumerate(_PROBE._EXPECTED_MARKERS.values())
        ]
        calls = [
            f'    %{index} = stablehlo.custom_call @"{_TARGET}"() : () -> tensor<1xbf16> loc(#loc{index})'
            for index in range(3)
        ]
        main_calls = [
            "    func.call @helper() : () -> ()" for _ in range(main_invocations)
        ]
        return "\n".join(
            [
                *definitions,
                "module {",
                "  func.func private @helper() {",
                *calls,
                "    return",
                "  }",
                "  func.func public @main() {",
                *main_calls,
                "    return",
                "  }",
                "}",
            ]
        )
    calls = [
        (
            f'  %{index} = bf16[1] custom-call(), custom_call_target="{_TARGET}", '
            f'op_name="{marker} query_start=0 query_size=512"'
        )
        for index, marker in enumerate(_PROBE._EXPECTED_MARKERS.values())
    ]
    main_calls = [
        f"  %call{index} = () call(), to_apply=%helper"
        for index in range(main_invocations)
    ]
    return "\n".join(
        [
            "HloModule helper_spoof",
            "helper {",
            *calls,
            "  ROOT %helper_root = () tuple()",
            "}",
            "ENTRY main {",
            *main_calls,
            "  ROOT %main_root = () tuple()",
            "}",
        ]
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_strict_ir_accepts_exact_three_calls_metadata_and_target(dialect):
    text = _stablehlo_calls() if dialect == "stablehlo" else _optimized_hlo_calls()
    summary = _PROBE._strict_vjp_ir_summary(text, dialect)
    assert summary["passed"] is True
    assert summary["custom_call_count"] == 3
    assert summary["marker_call_counts"] == {"forward": 1, "dq": 1, "dkdv": 1}
    assert summary["entry_call_ownership"]["direct_entry_custom_call_count"] == 3
    assert all(call["passed"] for call in summary["calls"])


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_raw_quote_first_lexing_ignores_escaped_quotes_braces_and_call_lookalikes(
    dialect,
):
    payload = (
        r"prefix \22 { custom-call( stablehlo.custom_call func.call ENTRY } suffix"
    )
    if dialect == "stablehlo":
        text = _stablehlo_calls().replace(
            "() : () -> tensor<1xbf16>",
            f'() {{backend_config = "{payload}"}} : () -> tensor<1xbf16>',
        )
    else:
        text = f'HloModule quoted_spoof, config="{payload}"\n' + _optimized_hlo_calls()
    summary = _PROBE._strict_vjp_ir_summary(text, dialect)
    assert summary["passed"] is True
    lexing = summary["quote_lexing_diagnostic"]
    assert lexing["syntax_was_lexed_before_local_escape_decoding"] is True
    assert lexing["raw_unquoted_custom_call_count"] == 3
    assert lexing["decoded_then_masked_custom_call_count_diagnostic_only"] != 3
    assert lexing["raw_mlir_hex_quote_escape_count"] > 0
    assert lexing["raw_quote_scan_passed"] is True
    serialized = json.dumps(summary)
    assert payload not in serialized
    assert "quoted_spoof" not in serialized


@pytest.mark.parametrize(
    "corrupt",
    (
        'HloModule broken, config="unterminated\n',
        'HloModule broken, config="bad \\q escape"\n',
    ),
)
def test_optimized_quote_scanner_fails_closed_on_unterminated_or_unknown_escape(
    corrupt,
):
    summary = _PROBE._strict_vjp_ir_summary(
        corrupt + _optimized_hlo_calls(), "optimized_hlo"
    )
    assert summary["passed"] is False
    ownership = summary["entry_call_ownership"]
    assert ownership["raw_quote_scan"]["passed"] is False
    assert ownership["checks"]["raw_quote_escapes_are_well_formed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize("main_invocations", (0, 2))
def test_strict_ir_rejects_unreachable_or_repeated_helper_owned_calls(
    dialect, main_invocations
):
    summary = _PROBE._strict_vjp_ir_summary(
        _helper_owned_calls(dialect, main_invocations=main_invocations), dialect
    )
    assert summary["custom_call_count"] == 3
    assert summary["entry_call_ownership"]["direct_entry_custom_call_count"] == 0
    helper = next(
        item
        for item in summary["entry_call_ownership"]["computations"]
        if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == (
        0 if main_invocations == 0 else ">1"
    )
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_helper_called_once_is_reported_once_but_runtime_direct_entry_gate_rejects(
    dialect,
):
    summary = _PROBE._strict_vjp_ir_summary(
        _helper_owned_calls(dialect, main_invocations=1), dialect
    )
    ownership = summary["entry_call_ownership"]
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["reachable_from_entry"] is True
    assert helper["saturated_entry_multiplicity"] == 1
    assert ownership["direct_entry_custom_call_count"] == 0
    assert summary["passed"] is False
    serialized = json.dumps(summary)
    assert "helper" not in serialized
    assert _TARGET not in serialized


def _cyclic_helper_owned_calls(dialect: str) -> str:
    text = _helper_owned_calls(dialect, main_invocations=1)
    if dialect == "stablehlo":
        return text.replace(
            "  func.func private @helper() {",
            "  func.func private @helper() {\n    func.call @helper() : () -> ()",
            1,
        )
    return text.replace(
        "helper {",
        "helper {\n  %recursive = () call(), to_apply=%helper",
        1,
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_reachable_recursive_helper_is_cycle_unknown_and_rejected(dialect):
    summary = _PROBE._strict_vjp_ir_summary(
        _cyclic_helper_owned_calls(dialect), dialect
    )
    ownership = summary["entry_call_ownership"]
    assert ownership["entry_multiplicity"]["cycle_detected"] is True
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == "unknown"
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_direct_custom_call_with_called_computation_is_unknown_and_rejected(dialect):
    if dialect == "stablehlo":
        text = (
            _stablehlo_calls()
            .replace(
                "module {",
                "module {\n  func.func private @helper() { return }",
                1,
            )
            .replace(
                f'@"{_TARGET}"()',
                f'@"{_TARGET}"() {{called_computations = [@helper]}}',
                1,
            )
        )
    else:
        lines = _optimized_hlo_calls().splitlines()
        lines[1] += ", called_computations={%helper}"
        text = "\n".join(
            [
                "HloModule called_custom",
                "helper {",
                "  ROOT %helper_root = () tuple()",
                "}",
                *lines,
            ]
        )
    summary = _PROBE._strict_vjp_ir_summary(text, dialect)
    ownership = summary["entry_call_ownership"]
    assert (
        ownership["calls"][0]["has_nonempty_or_malformed_called_computations"] is True
    )
    assert ownership["forbidden_container_count"] > 0
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == "unknown"
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "instruction",
    (
        "%wrapped = () conditional(), branch_computations={%helper}",
        "%wrapped = () while(), condition=%helper, body=%helper",
        "%wrapped = () async-start(), calls={%helper}",
        "%wrapped = () async-start(), calls=%helper",
        "%wrapped = () mystery(), to_apply=%helper",
    ),
)
def test_optimized_conditional_loop_async_and_unknown_wrappers_are_unknown_rejected(
    instruction,
):
    text = _helper_owned_calls("optimized_hlo", main_invocations=0).replace(
        "ENTRY main {", f"ENTRY main {{\n  {instruction}", 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["forbidden_container_count"] > 0
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == "unknown"
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize("suffix", ("x", "garbage", "e2", ".0"))
def test_strict_ir_rejects_metadata_numeric_suffix_spoofs(dialect, suffix):
    metadata = dict.fromkeys(
        _PROBE._EXPECTED_MARKERS,
        f"query_start=0{suffix} query_size=512{suffix}",
    )
    text = (
        _stablehlo_calls(metadata=metadata)
        if dialect == "stablehlo"
        else _optimized_hlo_calls(metadata=metadata)
    )
    assert _PROBE._strict_vjp_ir_summary(text, dialect)["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize(
    "failure", ("target", "marker", "extra", "while", "duplicate_metadata")
)
def test_strict_ir_rejects_target_marker_count_while_and_duplicate_metadata(
    dialect, failure
):
    target = "triton" if failure == "target" else _TARGET
    markers = dict(_PROBE._EXPECTED_MARKERS)
    metadata = dict.fromkeys(_PROBE._EXPECTED_MARKERS, "query_start=0 query_size=512")
    extra = ""
    if failure == "marker":
        markers["dq"] = "query_bounded_gqa_dq_q0_spoof"
    elif failure == "extra":
        extra = (
            f'  %9 = stablehlo.custom_call @"{_TARGET}"() : () -> tensor<1xbf16>'
            if dialect == "stablehlo"
            else f'  %9 = bf16[1] custom-call(), custom_call_target="{_TARGET}"'
        )
    elif failure == "while":
        extra = (
            "%loop = stablehlo.while() : () -> ()"
            if dialect == "stablehlo"
            else "%loop = bf16[1] while(%x)"
        )
    elif failure == "duplicate_metadata":
        metadata["forward"] += " query_start=0"
    text = (
        _stablehlo_calls(target=target, metadata=metadata, markers=markers, extra=extra)
        if dialect == "stablehlo"
        else _optimized_hlo_calls(
            target=target, metadata=metadata, markers=markers, extra=extra
        )
    )
    assert _PROBE._strict_vjp_ir_summary(text, dialect)["passed"] is False


def test_optimized_target_nested_only_in_backend_config_cannot_spoof_top_level_target():
    forged = _optimized_hlo_calls().replace(
        f'custom_call_target="{_TARGET}"',
        f'backend_config={{custom_call_target="{_TARGET}"}}',
    )
    summary = _PROBE._strict_vjp_ir_summary(forged, "optimized_hlo")
    assert summary["custom_call_count"] == 3
    assert all(call["target_count"] == 0 for call in summary["calls"])
    assert summary["passed"] is False


def test_structural_gate_requires_both_dialects():
    stable = _PROBE._strict_vjp_ir_summary(_stablehlo_calls(), "stablehlo")
    optimized = _PROBE._strict_vjp_ir_summary(_optimized_hlo_calls(), "optimized_hlo")
    assert _PROBE._structural_gate(stable, optimized)["passed"] is True
    assert _PROBE._structural_gate(stable)["passed"] is False


def test_memory_gate_requires_exact_promoted_signature_and_zero_alias():
    valid = {
        "available": True,
        "argument_size_in_bytes": 10_487_808,
        "output_size_in_bytes": 10_485_792,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 64 * 1024**2,
    }
    assert _PROBE._compiled_memory_gate(valid)["passed"] is True
    for name, value in (
        ("argument_size_in_bytes", 10_487_807),
        ("output_size_in_bytes", 10_485_791),
        ("alias_size_in_bytes", 1),
        ("temp_size_in_bytes", 64 * 1024**2 + 1),
    ):
        corrupt = dict(valid)
        corrupt[name] = value
        assert _PROBE._compiled_memory_gate(corrupt)["passed"] is False


class _FakeShape:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _CompiledFake:
    def __init__(self, state):
        self.state = state

    def as_text(self):
        return _optimized_hlo_calls()

    def memory_analysis(self):
        return SimpleNamespace(
            argument_size_in_bytes=10_487_808,
            output_size_in_bytes=10_485_792,
            alias_size_in_bytes=0,
            temp_size_in_bytes=2_130_688,
        )

    def __call__(self, *arguments):
        self.state["compiled_invocations"] += 1
        return (arguments[0], arguments[0], arguments[1], arguments[2])


class _LoweredFake:
    def __init__(self, state, compiled):
        self.state = state
        self.compiled = compiled

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        return _stablehlo_calls()

    def compile(self):
        self.state["compile_calls"] += 1
        return self.compiled


class _JitFake:
    def __init__(self, function, state, lowered):
        self.function = function
        self.state = state
        self.lowered = lowered

    def lower(self, *signature):
        self.state["lower_calls"] += 1
        self.state["signature"] = signature
        self.function(*signature)
        return self.lowered


class _JaxFake:
    ShapeDtypeStruct = _FakeShape

    def __init__(self, state, lowered):
        self.state = state
        self.lowered = lowered

    def jit(self, function):
        return _JitFake(function, self.state, self.lowered)

    def vjp(self, function, q, k, v):
        value = function(q, k, v)
        return value, lambda dout: (dout, k, v)


def test_compile_mock_uses_explicit_scale_three_calls_and_releases_without_invocation():
    state = {
        "lower_calls": 0,
        "compile_calls": 0,
        "compiled_invocations": 0,
        "api": [],
    }
    compiled = _CompiledFake(state)
    lowered = _LoweredFake(state, compiled)

    def api(*arguments, **keywords):
        state["api"].append((arguments, keywords))
        return arguments[0]

    counters = _PROBE._zero_counters()
    compiled_artifact, report = _PROBE._compile_vjp_artifact(
        _JaxFake(state, lowered),
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        api,
        lambda: dict(_CLEAN),
        counters,
        io.StringIO(),
    )
    checked = _PROBE._release_checked_vjp(compiled_artifact, report, counters)
    assert checked.proof["passed"] is True
    assert state["lower_calls"] == state["compile_calls"] == 1
    assert state["compiled_invocations"] == 0
    assert [item.shape for item in state["signature"]] == [
        (1, 512, 16, 256),
        (1, 512, 4, 256),
        (1, 512, 4, 256),
        (1, 512),
        (1, 512, 16, 256),
    ]
    assert state["api"][0][1] == {
        "scale": 0.09375,
        "query_chunk_size": 512,
        "block_q": 64,
        "block_k": 64,
        "backward_block_q": 32,
        "backward_block_k": 32,
        "interpret": False,
    }
    assert report["release_gate"]["exact_logical_dispatches"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
    }


def test_checked_capability_is_private_single_use_and_counts_one_executable():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    compiled = _CompiledFake(state)
    with pytest.raises(RuntimeError, match="without passed gates"):
        _PROBE._CheckedVjpExecutable(
            compiled, proof={"passed": True}, counters=counters, token=object()
        )
    checked = _PROBE._wrap_checked(compiled, {"passed": True}, counters)
    fake_jax = SimpleNamespace(block_until_ready=lambda value: value)
    arguments = ("q", "k", "v", "mask", "dout")
    result = checked.invoke(fake_jax, arguments, lambda: None)
    assert result == ("q", "q", "k", "v")
    assert state["compiled_invocations"] == 1
    assert counters["checked_executable_invocations"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        checked.invoke(fake_jax, arguments, lambda: None)


def test_bf16_rounded_reference_passes_and_corrupted_gradient_fails(exact_case):
    _inputs, _manifests, expected, _reference = exact_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)
    record = _PROBE._validate_candidate(
        np,
        actual,
        expected,
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
    )
    assert record["gates"]["promotion_passed"] is True
    assert all(record["gates"]["per_tensor_numerical_passed"].values())
    corrupted = list(actual)
    corrupted[2] = np.full_like(corrupted[2], ml_dtypes.bfloat16(2.0))
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            tuple(corrupted),
            expected,
            0.01,
            _PROBE._completed_counters(),
            io.StringIO(),
        )


def test_nonfinite_candidate_emits_json_safe_failure_before_failing(exact_case):
    _inputs, _manifests, expected, _reference = exact_case
    actual = [item.astype(ml_dtypes.bfloat16) for item in expected]
    actual[1].reshape(-1)[0] = ml_dtypes.bfloat16(float("nan"))
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            tuple(actual),
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
        )
    [record] = _records(output)
    assert record["record_type"] == "host_vjp_validation"
    assert record["gates"]["per_tensor_numerical_passed"]["dq"] is False
    assert record["metrics"]["dq"]["relative_l2"] is None
    assert record["metrics"]["dq"]["cosine"] is None


@pytest.mark.parametrize(
    "duration",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "posinf", "neginf"),
)
def test_nonfinite_candidate_duration_serializes_dispatch_then_fails_validation(
    duration, exact_case, monkeypatch
):
    _inputs, _manifests, expected, _reference = exact_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)

    class Compiled:
        def __call__(self, *_arguments):
            return actual

    counters = _PROBE._zero_counters()
    executable = _PROBE._wrap_checked(Compiled(), {"passed": True}, counters)
    clock = iter((123.0, 0.0, duration))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        _PROBE,
        "_journal_checkpoint",
        lambda _clean, _output, _stage, _counters: dict(_CLEAN),
    )
    output = io.StringIO()
    dispatched, measured = _PROBE._dispatch_candidate(
        SimpleNamespace(block_until_ready=lambda value: value),
        executable,
        ("q", "k", "v", "mask", "dout"),
        lambda: dict(_CLEAN),
        counters,
        output,
    )
    assert dispatched is actual
    assert not np.isfinite(measured)
    dispatch_records = _records(output)
    assert dispatch_records[-1]["record_type"] == "dispatch"
    assert dispatch_records[-1]["seconds"] is None

    with pytest.raises(RuntimeError, match="duration gates"):
        _PROBE._validate_candidate(np, dispatched, expected, measured, counters, output)
    records = _records(output)
    validation = records[-1]
    assert validation["record_type"] == "host_vjp_validation"
    assert validation["candidate_total_seconds"] is None
    assert validation["gates"]["safety_duration_passed"] is False
    assert validation["gates"]["promotion_duration_passed"] is False


def test_compile_diagnostic_destroys_unreleased_artifact_with_all_runtime_counters_zero(
    monkeypatch,
):
    state = {"invocations": 0, "destroyed": 0}

    class NeverCallable:
        def __call__(self, *_arguments):
            state["invocations"] += 1
            raise AssertionError("compile diagnostic invoked compiled executable")

        def __del__(self):
            state["destroyed"] += 1

    def compile_artifact(_jax, _jnp, _api, _clean, counters, _output):
        for name in (
            "lower_attempts",
            "lower_completions",
            "compile_attempts",
            "compile_completions",
        ):
            counters[name] += 1
        return NeverCallable(), {
            "structural_gate": {"passed": True},
            "compiled_memory_gate": {"passed": True},
            "release_gate": {"passed": True},
        }

    monkeypatch.setattr(_PROBE, "_compile_vjp_artifact", compile_artifact)
    for forbidden in (
        "_release_checked_vjp",
        "_construct_host_case",
        "_device_put_inputs",
        "_dispatch_candidate",
        "_device_get_candidate",
    ):
        monkeypatch.setattr(
            _PROBE,
            forbidden,
            lambda *_args, _name=forbidden, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"compile diagnostic reached {_name}")
            ),
        )
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    result = _PROBE._run_compile_diagnostic(
        object(),
        object(),
        lambda *_args, **_kwargs: None,
        lambda: dict(_CLEAN),
        counters,
        output,
    )
    gc.collect()
    assert result == 0
    assert state == {"invocations": 0, "destroyed": 1}
    assert counters == _PROBE._compile_diagnostic_completed_counters()
    for name in (
        "input_device_put_attempts",
        "input_device_put_completions",
        "candidate_attempts",
        "candidate_completions",
        "device_get_attempts",
        "device_get_completions",
        "lowered_callable_invocations",
        "checked_executable_invocations",
        "checked_capability_release_attempts",
        "checked_capability_release_completions",
        "host_reference_construction_attempts",
        "host_reference_construction_completions",
    ):
        assert counters[name] == 0
    [record] = _records(output)
    assert record["record_type"] == "compile_diagnostic_completed"
    assert record["status"] == "structure_and_memory_passed_no_release"
    assert record["checked_capability_created_or_released"] is False
    assert record["host_reference_constructed"] is False
    assert record["host_or_device_inputs_constructed"] is False
    assert record["executable_invoked"] is False
    assert record["device_output_retrieved"] is False


def test_fake_run_rocm_executes_one_put_one_checked_call_one_get_and_exact_counters(
    monkeypatch,
):
    state = {
        "compiled_invocations": 0,
        "device_put_calls": 0,
        "device_get_calls": 0,
        "journal_stages": [],
    }

    def compile_checked(_jax, _jnp, _api, _clean, counters, _output):
        for name in (
            "lower_attempts",
            "lower_completions",
            "compile_attempts",
            "compile_completions",
        ):
            counters[name] = 1
        return (
            _CompiledFake(state),
            {
                "release_gate": {"passed": True},
                "compiled_memory_gate": {"passed": True},
            },
        )

    def checkpoint(require_clean, output, stage, counters):
        assert require_clean() == _CLEAN
        state["journal_stages"].append(stage)
        _PROBE._emit(
            {
                "record_type": "journal_checkpoint",
                "stage": stage,
                "counters": dict(counters),
            },
            output,
        )
        return dict(_CLEAN)

    def validate(_np, actual, expected, seconds, counters, output):
        assert actual == ("q", "q", "k", "v")
        assert expected == (
            "expected-output",
            "expected-dq",
            "expected-dk",
            "expected-dv",
        )
        assert 0 <= seconds < 0.1
        assert counters == _PROBE._completed_counters()
        record = {"gates": {"promotion_passed": True}}
        _PROBE._emit({"record_type": "host_vjp_validation", **record}, output)
        return record

    class FakeJax:
        @staticmethod
        def block_until_ready(value):
            return value

        @staticmethod
        def device_put(value):
            state["device_put_calls"] += 1
            return value

        @staticmethod
        def device_get(value):
            state["device_get_calls"] += 1
            return value

    backend = SimpleNamespace(
        _backend_manifest=lambda *_args: {
            "platform_resolved": "gpu",
            "platform_family": "rocm",
            "visible_device_count": 1,
        }
    )
    monkeypatch.setattr(
        _PROBE,
        "_nonzero_probe",
        lambda: SimpleNamespace(_length_probe=lambda: backend),
    )
    monkeypatch.setattr(
        _PROBE, "_prove_command_buffers_disabled", lambda _env: {"passed": True}
    )
    monkeypatch.setattr(
        _PROBE,
        "_assert_gfx1100_drm",
        lambda: {"passed": True, "architecture": "gfx1100"},
    )
    monkeypatch.setattr(_PROBE, "_assert_kernel_binding", lambda _api: {"passed": True})
    monkeypatch.setattr(_PROBE, "_compile_vjp_artifact", compile_checked)
    monkeypatch.setattr(_PROBE, "_journal_checkpoint", checkpoint)
    monkeypatch.setattr(
        _PROBE,
        "_construct_host_case",
        lambda _np, _ml: (
            ("q", "k", "v", "mask", "dout"),
            [],
            ("expected-output", "expected-dq", "expected-dk", "expected-dv"),
            {"oracle": {"accelerator_used": False}},
        ),
    )
    monkeypatch.setattr(_PROBE, "_validate_candidate", validate)
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    result = _PROBE._run_rocm(
        output,
        lambda: dict(_CLEAN),
        counters,
        environment={},
        _dependencies=(
            FakeJax,
            object(),
            object(),
            object(),
            object(),
            object(),
            lambda *_args, **_kwargs: None,
        ),
    )
    assert result == 0
    assert counters == _PROBE._completed_counters()
    assert state["compiled_invocations"] == 1
    assert state["device_put_calls"] == state["device_get_calls"] == 1
    assert state["journal_stages"] == [
        "after_backend_initialization_attempt",
        "after_host_reference_construction",
        "after_explicit_input_device_put_attempt",
        "after_candidate_dispatch_attempt",
        "after_candidate_device_get_attempt",
        "after_host_validation",
    ]
    records = _records(output)
    assert records[-1]["record_type"] == "runtime_passed"
    assert records[-1]["counters"] == _PROBE._completed_counters()
    backend_record = next(
        item for item in records if item["record_type"] == "backend_ready"
    )
    assert backend_record["architecture_binding"]["architecture"] == "gfx1100"


def test_ast_proves_lazy_import_one_compile_one_vjp_and_one_checked_execution():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    roots = {
        alias.name.partition(".")[0]
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl"}.isdisjoint(roots)
    compile_source = inspect.getsource(_PROBE._compile_vjp_artifact)
    assert compile_source.count("jax.vjp(") == 1
    assert compile_source.count(".lower(") == 1
    assert compile_source.count("lowered.compile(") == 1
    assert (
        inspect.getsource(_PROBE._dispatch_candidate).count("executable.invoke(") == 1
    )
    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index(
        "import jax"
    )
    assert run_source.index("_assert_gfx1100_drm") < run_source.index("import jax")
    assert run_source.count("_device_put_inputs(") == 1
    assert run_source.count("_device_get_candidate(") == 1
    assert "if counters != _completed_counters():" in run_source
    diagnostic_source = inspect.getsource(_PROBE._run_compile_diagnostic)
    for forbidden in (
        "_wrap_checked",
        "_CheckedVjpExecutable",
        "_release_checked_vjp",
        "_construct_host_case",
        "_device_put_inputs",
        "_dispatch_candidate",
        "_device_get_candidate",
    ):
        assert forbidden not in diagnostic_source
    execute_source = inspect.getsource(_PROBE._execute)
    assert execute_source.index("_load_safety_helpers") < execute_source.index(
        "_configure_rocm_environment"
    )


def test_default_subprocess_refuses_without_importing_jax():
    program = f"""
import contextlib, importlib.util, io, json, sys
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('isolated_vjp_probe',{str(_PROBE_PATH)!r})
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
captured=io.StringIO()
with contextlib.redirect_stdout(captured):
    result=module.main([])
new=set(sys.modules)-before
bad=sorted(name for name in new if name in {{'jax','jaxlib'}} or name.startswith(('jax.','jaxlib.')))
print(json.dumps({{'result':result,'bad':bad,'records':[json.loads(line) for line in captured.getvalue().splitlines()]}}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_REPO)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["result"] == 0
    assert result["bad"] == []
    assert [item["record_type"] for item in result["records"]] == [
        "manifest",
        "refused",
    ]
