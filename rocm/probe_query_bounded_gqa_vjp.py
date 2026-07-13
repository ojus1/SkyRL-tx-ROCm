#!/usr/bin/env python3
"""Guarded exact-T512 full GQA VJP gate.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm requires ``--platform rocm --allow-gpu --output`` in a fresh process under
``profile_rocm.py``.  The exact BF16 B1/T512/Hq16/Hkv4/D256 forward and custom
VJP are lowered and compiled once with explicit scale 3/32.  A private
capability is released only after StableHLO and optimized HLO independently
prove exactly one direct entry-level q0 forward, q0 dQ, and q0 dK/dV logical
ROCm Triton call, canonical query metadata, no extra custom call, callable
container that can own or duplicate a custom call, or outer while, and exact
compiled-memory bounds.  Unrelated direct-entry fusion helpers are permitted
only when every nonentry computation contains zero custom calls.  The
capability is consumed by one invocation only.

``--compile-diagnostic`` is a separate compile-only authorization.  It emits
sanitized structural graphs and always destroys the unreleased compiled handle
before postflight.  That path cannot construct a host reference or device
inputs, create a checked capability, invoke the executable, or retrieve output.

Q/K/V and the output cotangent are independent deterministic nonzero BF16
PCG64 grids.  The default case remains all-valid.  The only padded case is
``--case valid385``: its int32 key mask is one through row 384 and zero after,
and its independently generated cotangent is set to bitwise positive zero
after row 384 to model loss masking.  Validation uses an independent host
NumPy FP32 causal-GQA forward/backward oracle.  It streams over
32-query/32-key tiles and recomputes probabilities, never retaining a complete
T-by-T matrix.  Output, dQ, dK, and dV are gated independently.  There is no
warmup, replay, second invocation, GPU reference/reduction, model call, or
backward-of-backward.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 512
_QUERY_HEADS = 16
_KV_HEADS = 4
_HEAD_DIM = 256
_GROUP_SIZE = _QUERY_HEADS // _KV_HEADS
_QUERY_CHUNK_SIZE = 512
_BLOCK_Q = 64
_BLOCK_K = 64
_BACKWARD_BLOCK_Q = 32
_BACKWARD_BLOCK_K = 32
_ORACLE_QUERY_TILE = 32
_ORACLE_KEY_TILE = 32
_ATTENTION_SCALE = 3 / 32
_INPUT_SEED = 20260713
_COTANGENT_SEED = 20260714
_GRID_DENOMINATOR = 128
_QKV_MAXIMUM_INTEGER_MAGNITUDE = 96
_COTANGENT_MAXIMUM_INTEGER_MAGNITUDE = 48
_ALL_VALID_CASE = "all_valid"
_VALID385_CASE = "valid385"
_CASE_VALID_TOKENS = {_ALL_VALID_CASE: 512, _VALID385_CASE: 385}
_EXPECTED_ARGUMENT_BYTES = 10_487_808
_EXPECTED_OUTPUT_BYTES = 10_485_792
_EXPECTED_ALIAS_BYTES = 0
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MIN_REFERENCE_NORM = 1e-8
_MAX_CANDIDATE_SECONDS = 0.1
_MAX_PROMOTION_CANDIDATE_SECONDS = 0.075
_EXACT_TARGET = "__gpu$xla.gpu.triton"
_EXPECTED_MARKERS = {
    "forward": "query_bounded_gqa_forward_q0",
    "dq": "query_bounded_gqa_dq_q0",
    "dkdv": "query_bounded_gqa_dkdv_q0",
}
_EXPECTED_COMPILE_PROBE_SOURCE_SHA256 = (
    "bf01187101d20362072c96f70fbb80b2f8eed88fa55e60cbf479abba6db012a2"
)
_EXPECTED_NONZERO_PROBE_SOURCE_SHA256 = (
    "999e027d4cc35a8d59cc294020f8865036f8fb817a847ac38f96e36b597f74ac"
)
_EXPECTED_KERNEL_SOURCE_SHA256 = (
    "51e2fd91eb270f7b25ecdd117d7f06aa48a8e4af282a5a7e5e6b4c2a25dc52c9"
)
_EXPECTED_INPUT_SHA256 = {
    "q": "ff889e094bfb7ce5e55446f1fff3956a747eed7986e9c30d170f7c4433b9bda7",
    "k": "e5d2258530fcb373b981167ba116b43306bb424f565bbf2da7f5c55810369978",
    "v": "26f13a090d34805fb6fc1ea002c65e7d7cdb9952b8a582cd132246fe54420699",
    "key_mask": "6323b30c3d5f9b893f1133983aa3761cef653959de5a6e4f8e798c358bd226e1",
    "dout": "23b69593d1dde3c5a9bea2d5679fe78891e5d96d083e31b5fc4fa1460d3c0622",
}
_EXPECTED_REFERENCE_SHA256 = {
    "output": "273c1e5694bc21b03798ccd73b8c8be669218d41c82e1b23236c2e3a4bea506f",
    "dq": "936167f7fcc92a5bc8251ca4c332f80fae74ab36c98ca552d68e16da651702d5",
    "dk": "04e64911be655c3f1bacf2d3d091be301df06d2a98d534267e4ebde3b74d24e6",
    "dv": "9c4f5d999e7548e769dad32950b9177a939b114f06436b75ab4acb38c9e73e83",
}
_EXPECTED_REFERENCE_NORMS = {
    "output": 75.4267547716931,
    "dq": 9.212431174660864,
    "dk": 9.257514343887854,
    "dv": 38.029129972931884,
}
_EXPECTED_ORACLE_SCRATCH_BYTES = 323_072
_EXPECTED_MAXIMUM_ABSOLUTE_VALID_LOGIT = 1.4235591888427734
_EXPECTED_VALID385_INPUT_SHA256 = {
    "q": "ff889e094bfb7ce5e55446f1fff3956a747eed7986e9c30d170f7c4433b9bda7",
    "k": "e5d2258530fcb373b981167ba116b43306bb424f565bbf2da7f5c55810369978",
    "v": "26f13a090d34805fb6fc1ea002c65e7d7cdb9952b8a582cd132246fe54420699",
    "key_mask": "7fe610e6a15b1f19c3266fff71c6c63030a974bf986266a719a388f76f7b9299",
    "dout": "2641e35a3b1dcdd277a6aa44ccc341552112617b77f3d12585eac15cd74c905b",
}
_EXPECTED_VALID385_REFERENCE_SHA256 = {
    "output": "7e488e1ef314cbc513714ed4914cecc398d184d46e52bef14eb9cf513c45d8b1",
    "dq": "f995a2ec2da893d24f3a23fb71e984907ce81b6759983a4cecc95dbc94df29c5",
    "dk": "813dc721ff7931a15e4d5b0cc61f493e6cc222bedf813761f681b481fe0bb291",
    "dv": "10a42a2ae5ee8c8d99ee77919aeec1f1598d30c18e1fc065f2631d5833c6426c",
}
_EXPECTED_VALID385_REFERENCE_NORMS = {
    "output": 75.66161081261592,
    "dq": 8.937369297138494,
    "dk": 8.988992969708956,
    "dv": 37.09246081216146,
}
_EXPECTED_VALID385_MAXIMUM_ABSOLUTE_VALID_LOGIT = 1.4235591888427734
_EXPECTED_VALID385_SENSITIVITY_SHA256 = {
    "ignored_key_mask_output": "273c1e5694bc21b03798ccd73b8c8be669218d41c82e1b23236c2e3a4bea506f",
    "wrong_valid384_output": "2ad3945dcc49b1b4803bc0aca92dd4b9b48230c70f8730422a1a1c0ec006babe",
    "wrong_valid386_output": "f05bf273961a86119e7ef23cc29316b65d252954ccd547c72bc523a7efed7e3b",
    "ignored_loss_mask_dq": "51b55482a9effa4047952d143380f0f731882af3a94b34db88854669ab26235e",
    "ignored_loss_mask_dk": "fa41910c6e79e288575fc590306f80753b76a04656c0aa56d3356b5560d25ef6",
    "ignored_loss_mask_dv": "7e27fb07e7e521ae28ec7a1b4a3a9e50826f83592705511db1c810a7460bdd12",
}
_EXPECTED_VALID385_SENSITIVITY_METRICS = {
    "ignored_key_mask_affected": {
        "relative_l2": 0.3657309920526969,
        "cosine": 0.9307311482256659,
        "max_abs": 0.053003087639808655,
    },
    "ignored_key_mask_boundary_384_wrong_valid384": {
        "relative_l2": 0.051949241238546565,
        "cosine": 0.998651428667888,
        "max_abs": 0.00296160951256752,
    },
    "ignored_key_mask_boundary_385_wrong_valid386": {
        "relative_l2": 0.05737496052545676,
        "cosine": 0.9983563895485048,
        "max_abs": 0.004151057451963425,
    },
    "ignored_key_mask_boundary_511_ignored_key_mask": {
        "relative_l2": 0.49378035447136176,
        "cosine": 0.869597168664908,
        "max_abs": 0.053003087639808655,
    },
    "ignored_loss_mask_dq": {
        "relative_l2": 0.26905400487417747,
        "cosine": 0.9656585954892194,
        "max_abs": 0.016500195488333702,
    },
    "ignored_loss_mask_dk": {
        "relative_l2": 0.2670853309515711,
        "cosine": 0.9660999339963957,
        "max_abs": 0.01844042629818432,
    },
    "ignored_loss_mask_dv": {
        "relative_l2": 0.23070081049708538,
        "cosine": 0.9745800453229094,
        "max_abs": 0.05243309319484979,
    },
}
_EXPECTED_AMD_PCI_DEVICE_ID = "0x744c"
_EXPECTED_GPU_ARCHITECTURE = "gfx1100"
_IR_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
_CHECKED_CAPABILITY_TOKEN = object()
_JOURNAL_STAGES = frozenset(
    {
        "after_backend_initialization_attempt",
        "after_vjp_lower_attempt",
        "after_vjp_compile_attempt",
        "after_host_reference_construction",
        "after_explicit_input_device_put_attempt",
        "after_candidate_dispatch_attempt",
        "after_candidate_device_get_attempt",
        "after_host_validation",
    }
)
_COMPILE_GPU_WORK_CAVEAT = (
    "lowered.compile may dispatch bounded GPU autotuning/profiling work; the "
    "compiled VJP executable remains inaccessible until exact structural and "
    "compiled-memory gates pass, and compile-diagnostic mode never releases it"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _redacted_message_summary(error: BaseException) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    return {
        "message_redacted": True,
        "message_utf8_bytes": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _json_safe_duration(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _source_files() -> dict[str, Path]:
    repo = _repo_root()
    return {
        "probe_source_sha256": Path(__file__),
        "delegated_compile_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_compile.py",
        "delegated_nonzero_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_nonzero_scale.py",
        "query_bounded_gqa_kernel_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "query_bounded_gqa.py",
        "delegated_safety_helper_source_sha256": repo / "rocm" / "amdgpu_safety.py",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in _source_files().items()}


def _compile_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_compile
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_compile  # type: ignore[no-redef]

    return probe_query_bounded_gqa_compile


def _nonzero_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_nonzero_scale
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_nonzero_scale  # type: ignore[no-redef]

    return probe_query_bounded_gqa_nonzero_scale


def _assert_static_source_bindings() -> dict[str, Any]:
    files = _source_files()
    compile_probe = _compile_probe()
    nonzero = _nonzero_probe()
    expected_compile = files["delegated_compile_probe_source_sha256"].resolve()
    expected_nonzero = files["delegated_nonzero_probe_source_sha256"].resolve()
    if (
        not isinstance(getattr(compile_probe, "__file__", None), str)
        or Path(compile_probe.__file__).resolve() != expected_compile
    ):
        raise RuntimeError(
            "compile helper did not resolve to the exact repository source"
        )
    if (
        not isinstance(getattr(nonzero, "__file__", None), str)
        or Path(nonzero.__file__).resolve() != expected_nonzero
    ):
        raise RuntimeError(
            "nonzero helper did not resolve to the exact repository source"
        )
    hashes = {
        "delegated_compile_probe_source_sha256": _file_sha256(expected_compile),
        "delegated_nonzero_probe_source_sha256": _file_sha256(expected_nonzero),
        "query_bounded_gqa_kernel_source_sha256": _file_sha256(
            files["query_bounded_gqa_kernel_source_sha256"].resolve()
        ),
    }
    expected = {
        "delegated_compile_probe_source_sha256": _EXPECTED_COMPILE_PROBE_SOURCE_SHA256,
        "delegated_nonzero_probe_source_sha256": _EXPECTED_NONZERO_PROBE_SOURCE_SHA256,
        "query_bounded_gqa_kernel_source_sha256": _EXPECTED_KERNEL_SOURCE_SHA256,
    }
    if hashes != expected:
        raise RuntimeError("VJP delegated source SHA256 pin changed")
    delegated = nonzero._assert_static_source_bindings()
    if delegated.get("passed") is not True:
        raise RuntimeError("delegated promoted source binding did not pass")
    return {
        "passed": True,
        "compile_helper_resolved_file_matches_expected": True,
        "nonzero_helper_resolved_file_matches_expected": True,
        "delegated_source_binding": delegated,
        **hashes,
    }


def _assert_kernel_binding(api: Any) -> dict[str, Any]:
    expected_file = _source_files()["query_bounded_gqa_kernel_source_sha256"].resolve()
    loaded_file = inspect.getsourcefile(api)
    if loaded_file is None or Path(loaded_file).resolve() != expected_file:
        raise RuntimeError("VJP API did not bind to the exact repository kernel source")
    if _file_sha256(expected_file) != _EXPECTED_KERNEL_SOURCE_SHA256:
        raise RuntimeError("loaded VJP kernel source SHA256 changed")
    signature = inspect.signature(api)
    parameters = signature.parameters
    exact = (
        list(parameters)[:4] == ["q", "k", "v", "key_mask"]
        and parameters["scale"].default is None
        and parameters["query_chunk_size"].default == _QUERY_CHUNK_SIZE
        and parameters["block_q"].default == _BLOCK_Q
        and parameters["block_k"].default == _BLOCK_K
        and parameters["backward_block_q"].default == _BACKWARD_BLOCK_Q
        and parameters["backward_block_k"].default == _BACKWARD_BLOCK_K
        and parameters["interpret"].default is False
    )
    if not exact:
        raise RuntimeError("VJP API signature or defaults changed")
    return {
        "passed": True,
        "resolved_file_matches_expected": True,
        "source_sha256": _EXPECTED_KERNEL_SOURCE_SHA256,
        "signature_semantics_exact": True,
    }


def _assert_gfx1100_drm(
    drm_root: Path = Path("/sys/class/drm"),
) -> dict[str, Any]:
    """Bind the sole visible AMD card to this RX 7900 XTX PCI identity."""
    amd_cards: list[tuple[str, str]] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        try:
            vendor = (card / "device" / "vendor").read_text().strip().lower()
            device = (card / "device" / "device").read_text().strip().lower()
        except OSError:
            continue
        if vendor == "0x1002":
            amd_cards.append((card.name, device))
    if len(amd_cards) != 1 or amd_cards[0][1] != _EXPECTED_AMD_PCI_DEVICE_ID:
        raise RuntimeError(
            "VJP gate requires the sole AMD DRM card to be the pinned gfx1100 device"
        )
    return {
        "passed": True,
        "architecture": _EXPECTED_GPU_ARCHITECTURE,
        "amd_pci_vendor_id": "0x1002",
        "amd_pci_device_id": _EXPECTED_AMD_PCI_DEVICE_ID,
        "sole_amd_drm_card": amd_cards[0][0],
    }


def _zero_counters() -> dict[str, int]:
    return {
        "lower_attempts": 0,
        "lower_completions": 0,
        "compile_attempts": 0,
        "compile_completions": 0,
        "input_device_put_attempts": 0,
        "input_device_put_completions": 0,
        "candidate_attempts": 0,
        "candidate_completions": 0,
        "device_get_attempts": 0,
        "device_get_completions": 0,
        "lowered_callable_invocations": 0,
        "checked_executable_invocations": 0,
        "checked_capability_release_attempts": 0,
        "checked_capability_release_completions": 0,
        "host_reference_construction_attempts": 0,
        "host_reference_construction_completions": 0,
    }


def _completed_counters() -> dict[str, int]:
    result = _zero_counters()
    for name in (
        "lower_attempts",
        "lower_completions",
        "compile_attempts",
        "compile_completions",
        "input_device_put_attempts",
        "input_device_put_completions",
        "candidate_attempts",
        "candidate_completions",
        "device_get_attempts",
        "device_get_completions",
        "checked_executable_invocations",
        "checked_capability_release_attempts",
        "checked_capability_release_completions",
        "host_reference_construction_attempts",
        "host_reference_construction_completions",
    ):
        result[name] = 1
    return result


def _compile_diagnostic_completed_counters() -> dict[str, int]:
    result = _zero_counters()
    for name in (
        "lower_attempts",
        "lower_completions",
        "compile_attempts",
        "compile_completions",
    ):
        result[name] = 1
    return result


def _normalize_case(case: str) -> str:
    if case not in _CASE_VALID_TOKENS:
        raise RuntimeError("VJP case is outside the exact closed case set")
    return case


def _exact_contract(case: str = _ALL_VALID_CASE) -> dict[str, Any]:
    case = _normalize_case(case)
    valid_tokens = _CASE_VALID_TOKENS[case]
    q_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    contract = {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": (
            "query_bounded_gqa_t512_forward_and_full_vjp"
            if case == _ALL_VALID_CASE
            else "query_bounded_gqa_t512_valid385_loss_masked_forward_and_full_vjp"
        ),
        "gpu_architecture": _EXPECTED_GPU_ARCHITECTURE,
        "gpu_pci_device_id": _EXPECTED_AMD_PCI_DEVICE_ID,
        "inputs": [
            {"name": "q", "shape": q_shape, "dtype": "bfloat16"},
            {"name": "k", "shape": kv_shape, "dtype": "bfloat16"},
            {"name": "v", "shape": kv_shape, "dtype": "bfloat16"},
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, _SEQUENCE_LENGTH],
                "dtype": "int32",
                "value": (
                    "all_ones"
                    if case == _ALL_VALID_CASE
                    else "ones_before_385_zeros_at_and_after_385"
                ),
            },
            {
                "name": "dout",
                "shape": q_shape,
                "dtype": "bfloat16",
                "value": (
                    "independent_host_pcg64_nonzero_signed_grid"
                    if case == _ALL_VALID_CASE
                    else (
                        "independent_host_pcg64_nonzero_signed_grid_then_"
                        "bitwise_positive_zero_at_and_after_row_385"
                    )
                ),
            },
        ],
        "outputs": [
            {"name": "output", "shape": q_shape, "dtype": "bfloat16"},
            {"name": "dq", "shape": q_shape, "dtype": "bfloat16"},
            {"name": "dk", "shape": kv_shape, "dtype": "bfloat16"},
            {"name": "dv", "shape": kv_shape, "dtype": "bfloat16"},
        ],
        "scale": _ATTENTION_SCALE,
        "scale_exact_fraction": "3/32",
        "tiles": {
            "query_chunk_size": _QUERY_CHUNK_SIZE,
            "block_q": _BLOCK_Q,
            "block_k": _BLOCK_K,
            "backward_block_q": _BACKWARD_BLOCK_Q,
            "backward_block_k": _BACKWARD_BLOCK_K,
        },
        "compile_gate": {
            "required_dialects": ["stablehlo", "optimized_hlo"],
            "exact_custom_calls": dict.fromkeys(_EXPECTED_MARKERS, 1),
            "exact_total_custom_calls": 3,
            "sole_target_is_exact_rocm_triton": True,
            "sole_target_sha256": hashlib.sha256(
                _EXACT_TARGET.encode("utf-8")
            ).hexdigest(),
            "exact_query_start_metadata_per_call": 0,
            "exact_query_size_metadata_per_call": 512,
            "all_calls_directly_owned_by_sole_entry_computation": True,
            "no_container_can_own_or_duplicate_a_custom_call": True,
            "independent_zero_custom_call_entry_fusions_allowed": True,
            "no_outer_while": True,
            "exact_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
            "exact_output_bytes": _EXPECTED_OUTPUT_BYTES,
            "exact_alias_bytes": _EXPECTED_ALIAS_BYTES,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        },
        "execution_contract": {
            "lower_calls": 1,
            "compile_calls": 1,
            "input_tuple_device_put_calls": 1,
            "checked_executable_invocations": 1,
            "device_get_calls": 1,
            "logical_internal_dispatches": {
                "forward": 1,
                "dq": 1,
                "dkdv": 1,
                "total": 3,
            },
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "second_vjp_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_invocations": 0,
        },
        "reference": {
            "location": "host_numpy_only",
            "dtype": "float32",
            "algorithm": "three-pass tiled causal softmax forward and analytic VJP",
            "query_tile": _ORACLE_QUERY_TILE,
            "key_tile": _ORACLE_KEY_TILE,
            "full_t_by_t_matrix_constructed": False,
            "equations": {
                "delta": "row_sum(output * dout)",
                "dscores": "probability * (dout @ V.T - delta) * scale",
                "dq": "dscores @ K",
                "dk": "dscores.T @ Q grouped over query heads",
                "dv": "probability.T @ dout grouped over query heads",
            },
        },
        "numerical_gate_per_tensor": {
            "finite_required": True,
            "minimum_reference_l2_norm": _MIN_REFERENCE_NORM,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
        },
        "candidate_total_seconds_strictly_below": _MAX_CANDIDATE_SECONDS,
        "promotion_total_seconds_strictly_below": (_MAX_PROMOTION_CANDIDATE_SECONDS),
    }
    if case == _VALID385_CASE:
        contract.update(
            {
                "case": case,
                "valid_tokens": valid_tokens,
                "padding_side": "right",
                "loss_mask_applied_to_host_dout_before_device_put": True,
                "padded_zero_gates": {
                    "dq_query_rows_at_and_after_385_actual_finite_numeric_exact_zero": True,
                    "dk_key_rows_at_and_after_385_actual_finite_numeric_exact_zero": True,
                    "dv_key_rows_at_and_after_385_actual_finite_numeric_exact_zero": True,
                    "deterministic_fp32_reference_tails_bitwise_positive_zero": True,
                    "candidate_negative_zero_sign_bits_accepted": True,
                },
                "affected_forward_output_rows_numerically_gated": [385, 512],
                "individually_gated_boundary_output_rows": [384, 385, 511],
                "sensitivity_controls": {
                    "ignored_key_mask": "all-valid wrong-mask output rows 385:512 must fail the affected-row numerical gate",
                    "ignored_loss_mask": "unmasked-dout wrong gradient must fail the padded dQ bitwise-zero gate and full-gradient numerical gate",
                },
            }
        )
    return contract


def _abstract_contract() -> dict[str, Any]:
    return {
        "operation": "query_bounded_gqa_t512_full_vjp_refusal",
        "exact_logical_internal_calls": ["forward_q0", "dq_q0", "dkdv_q0"],
        "gpu_work_authorized": False,
    }


def _compile_diagnostic_contract(case: str = _ALL_VALID_CASE) -> dict[str, Any]:
    exact = _exact_contract(case)
    return {
        "operation": (
            "query_bounded_gqa_t512_full_vjp_compile_diagnostic"
            if case == _ALL_VALID_CASE
            else "query_bounded_gqa_t512_valid385_loss_masked_full_vjp_compile_diagnostic"
        ),
        "model_family": exact["model_family"],
        "gpu_architecture": exact["gpu_architecture"],
        "gpu_pci_device_id": exact["gpu_pci_device_id"],
        "inputs": exact["inputs"],
        "outputs": exact["outputs"],
        "scale": exact["scale"],
        "scale_exact_fraction": exact["scale_exact_fraction"],
        "tiles": exact["tiles"],
        "compile_gate": exact["compile_gate"],
        "checked_capability_creation_or_release_authorized": False,
        "host_reference_construction_authorized": False,
        "host_or_device_input_construction_authorized": False,
        "executable_invocation_authorized": False,
        "device_get_authorized": False,
        "always_stop_after_compile_and_postflight": True,
        "raw_ir_emitted": False,
        "case": case,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help="refusal-only by default; guarded ROCm work requires rocm plus --allow-gpu",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help=(
            "acknowledge one guarded compile and, unless --compile-diagnostic is "
            "set, one checked full-VJP invocation"
        ),
    )
    parser.add_argument(
        "--compile-diagnostic",
        action="store_true",
        help=(
            "compile and emit sanitized structural evidence only; never release "
            "or invoke the executable"
        ),
    )
    parser.add_argument(
        "--case",
        choices=tuple(_CASE_VALID_TOKENS),
        default=_ALL_VALID_CASE,
        help="closed host-input case set; default all_valid, or exact padded valid385",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive mode-0600 JSONL artifact (required for ROCm)",
    )
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error(
            "--platform rocm requires the explicit --allow-gpu acknowledgement"
        )
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform != "rocm" and args.compile_diagnostic:
        parser.error("--compile-diagnostic is only valid with --platform rocm")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a private JSONL artifact")
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite existing output")
    return args


def _configure_rocm_environment() -> dict[str, str | None]:
    return _nonzero_probe()._configure_rocm_environment()


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    return _nonzero_probe()._environment_manifest(environment)


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    return _nonzero_probe()._prove_command_buffers_disabled(environment)


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _nonzero_probe()._load_safety_helpers()


def _safety_binding_manifest(helpers: tuple[Any, Any]) -> dict[str, Any]:
    return _nonzero_probe()._safety_binding_manifest(helpers)


def _open_exclusive_output(path: Path) -> TextIO:
    return _nonzero_probe()._open_exclusive_output(path)


def _assert_fresh_accelerator_process() -> None:
    _nonzero_probe()._assert_fresh_accelerator_process()


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    return _nonzero_probe()._public_clean_safety(safety, stage)


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    return _nonzero_probe()._public_safety_preflight(safety)


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing an undeclared VJP journal stage")
    safety = _public_clean_safety(require_clean_boot(), stage)
    _emit(
        {
            "record_type": "journal_checkpoint",
            "timestamp": _utc_now(),
            "stage": stage,
            "safety": safety,
            "counters": dict(counters),
        },
        output,
    )
    return safety


def _decode_ir(text: str) -> str:
    return _nonzero_probe()._decode_local_mlir_hex_escapes(text)


def _raw_ir_quote_scan(text: str) -> dict[str, Any]:
    """Mask raw quoted payloads without decoding their escape syntax."""
    result = list(text)
    quoted = False
    quote_delimiters = 0
    escape_count = 0
    invalid_escape_count = 0
    index = 0
    while index < len(text):
        character = text[index]
        if quoted:
            if character != "\n":
                result[index] = " "
            if character == "\\":
                escape_count += 1
                if index + 2 < len(text) and re.fullmatch(
                    r"[0-9A-Fa-f]{2}", text[index + 1 : index + 3]
                ):
                    result[index + 1] = " "
                    result[index + 2] = " "
                    index += 2
                elif index + 1 < len(text) and text[index + 1] in (
                    '"\\/abfnrtv?01234567xXuU'
                ):
                    if text[index + 1] != "\n":
                        result[index + 1] = " "
                    index += 1
                else:
                    invalid_escape_count += 1
            elif character == '"':
                quoted = False
                quote_delimiters += 1
        elif character == '"':
            quoted = True
            quote_delimiters += 1
            result[index] = " "
        index += 1
    return {
        "masked": "".join(result),
        "quote_delimiter_count": quote_delimiters,
        "escape_count": escape_count,
        "invalid_escape_count": invalid_escape_count,
        "unterminated_quote": quoted,
        "passed": not quoted and invalid_escape_count == 0,
    }


def _mask_ir_quoted_content(text: str) -> str:
    return str(_raw_ir_quote_scan(text)["masked"])


def _symbol_sha256(symbol: str) -> str:
    return hashlib.sha256(symbol.encode("utf-8")).hexdigest()


def _saturated_entry_multiplicity(
    nodes: set[str],
    entry: str | None,
    raw_edges: list[tuple[str, str, str]],
) -> dict[str, Any]:
    edges = [
        (caller, callee, factor)
        for caller, callee, factor in raw_edges
        if caller in nodes and callee in nodes and factor in {"one", "unknown"}
    ]
    adjacency = {node: [] for node in nodes}
    indegree = dict.fromkeys(nodes, 0)
    for caller, callee, factor in edges:
        adjacency[caller].append((callee, factor))
        indegree[callee] += 1
    queue = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for callee, _factor in adjacency[node]:
            indegree[callee] -= 1
            if indegree[callee] == 0:
                queue.append(callee)
                queue.sort()
    cycle = len(order) != len(nodes)
    reachable: set[str] = set()
    if entry in nodes:
        pending = [entry]
        while pending:
            node = pending.pop()
            if node in reachable:
                continue
            reachable.add(node)
            pending.extend(callee for callee, _factor in adjacency[node])
    multiplicity = dict.fromkeys(nodes, 0)
    unknown_multiplicity = dict.fromkeys(nodes, False)
    if entry in nodes:
        multiplicity[entry] = 1
    if not cycle:
        for node in order:
            for callee, factor in adjacency[node]:
                if unknown_multiplicity[node] or (
                    factor == "unknown" and multiplicity[node] > 0
                ):
                    unknown_multiplicity[callee] = True
                elif factor == "one":
                    multiplicity[callee] = min(
                        2, multiplicity[callee] + multiplicity[node]
                    )
    return {
        "cycle_detected": cycle,
        "unknown_edge_count": len(raw_edges) - len(edges),
        "unknown_factor_edge_count": sum(
            factor == "unknown" for _caller, _callee, factor in edges
        ),
        "nodes": {
            node: {
                "reachable_from_entry": node in reachable,
                "saturated_entry_multiplicity": (
                    "unknown"
                    if (cycle and node in reachable) or unknown_multiplicity[node]
                    else ">1"
                    if multiplicity[node] > 1
                    else multiplicity[node]
                ),
            }
            for node in sorted(nodes)
        },
    }


def _raw_custom_call_blocks(text: str, dialect: str) -> list[str]:
    """Extract only unquoted custom-call instructions while preserving raw text."""
    masked = _mask_ir_quoted_content(text)
    raw_lines = text.splitlines()
    masked_lines = masked.splitlines()
    if len(raw_lines) != len(masked_lines):
        raise RuntimeError("raw IR quote masking changed the line count")
    if dialect == "stablehlo":
        start_pattern = re.compile(
            r"^(?P<indent>\s*)%[^=]+?=\s*stablehlo\.custom_call\b"
        )
        boundary_pattern = re.compile(
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|"
            r"(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start_pattern = re.compile(
            r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\("
        )
        boundary_pattern = re.compile(r"^\s*(?:(?:ROOT\s+)?[^=]+?=|}\s*$)")
    else:
        raise ValueError("unsupported VJP IR dialect")
    blocks: list[str] = []
    index = 0
    while index < len(masked_lines):
        start = start_pattern.search(masked_lines[index])
        if start is None:
            index += 1
            continue
        base_indent = len(start.group("indent").expandtabs())
        block_lines = [raw_lines[index]]
        index += 1
        while index < len(masked_lines):
            candidate = masked_lines[index]
            candidate_indent = len(candidate) - len(candidate.lstrip(" \t"))
            if (
                candidate.strip()
                and candidate_indent <= base_indent
                and boundary_pattern.match(candidate)
            ):
                break
            block_lines.append(raw_lines[index])
            index += 1
        blocks.append("\n".join(block_lines))
    return blocks


def _isolated_custom_call_target_occurrences(block: str, dialect: str) -> list[str]:
    if dialect == "stablehlo":
        return [
            match.group(1) or match.group(2)
            for match in re.finditer(
                r'\bstablehlo\.custom_call\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))',
                block,
            )
        ]
    if dialect == "optimized_hlo":
        masked = _mask_ir_quoted_content(block)
        opcode = re.search(r"\bcustom-call\s*\(", masked)
        if opcode is None:
            return []
        opening = masked.find("(", opcode.start(), opcode.end())
        depth = 0
        closing = None
        for index in range(opening, len(masked)):
            if masked[index] == "(":
                depth += 1
            elif masked[index] == ")":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
                if depth < 0:
                    return []
        if closing is None:
            return []
        key = re.compile(r"\bcustom_call_target\s*=")
        nesting = {"(": 0, "[": 0, "{": 0}
        pairs = {")": "(", "]": "[", "}": "{"}
        occurrences: list[str] = []
        index = closing + 1
        while index < len(masked):
            character = masked[index]
            if character in nesting:
                nesting[character] += 1
            elif character in pairs:
                opening_character = pairs[character]
                nesting[opening_character] -= 1
                if nesting[opening_character] < 0:
                    return ["<malformed-target-structure>"]
            elif all(value == 0 for value in nesting.values()):
                match = key.match(masked, index)
                if match is not None:
                    value_start = match.end()
                    while value_start < len(block) and block[value_start].isspace():
                        value_start += 1
                    if value_start >= len(block) or block[value_start] != '"':
                        occurrences.append("<malformed-target-value>")
                        index = match.end()
                        continue
                    value_stop = value_start + 1
                    escaped = False
                    while value_stop < len(block):
                        value_character = block[value_stop]
                        if escaped:
                            escaped = False
                        elif value_character == "\\":
                            escaped = True
                        elif value_character == '"':
                            break
                        value_stop += 1
                    if value_stop >= len(block):
                        occurrences.append("<malformed-target-value>")
                    else:
                        occurrences.append(
                            _decode_ir(block[value_start + 1 : value_stop])
                        )
                    index = value_stop
            index += 1
        if any(nesting.values()):
            return ["<malformed-target-structure>"]
        return occurrences
    raise ValueError("unsupported VJP IR dialect")


def _top_level_hlo_computation_references(instruction: str) -> dict[str, Any]:
    """Parse computation references only from an instruction's depth-zero tail."""
    masked = _mask_ir_quoted_content(instruction)
    assignment = masked.find("=")
    if assignment < 0:
        return {"attributes": [], "malformed_count": 1, "passed": False}
    opcode = re.search(
        r"(?P<opcode>[A-Za-z_][A-Za-z0-9_.-]*)\s*\(",
        masked[assignment + 1 :],
    )
    if opcode is None:
        return {"attributes": [], "malformed_count": 1, "passed": False}
    opcode_start = assignment + 1 + opcode.start()
    opening = masked.find("(", opcode_start, assignment + 1 + opcode.end())
    depth = 0
    closing = None
    for index in range(opening, len(masked)):
        if masked[index] == "(":
            depth += 1
        elif masked[index] == ")":
            depth -= 1
            if depth == 0:
                closing = index
                break
            if depth < 0:
                break
    if closing is None:
        return {"attributes": [], "malformed_count": 1, "passed": False}

    key_pattern = re.compile(
        r"\b(?P<key>to_apply|calls|called_computations|branch_computations|"
        r"condition|body)\s*="
    )
    symbol_pattern = re.compile(r"%?[A-Za-z_][A-Za-z0-9_.-]*")
    nesting = {"(": 0, "[": 0, "{": 0}
    closing_to_opening = {")": "(", "]": "[", "}": "{"}
    attributes: list[dict[str, Any]] = []
    malformed_count = 0
    index = closing + 1
    while index < len(masked):
        character = masked[index]
        if character in nesting:
            nesting[character] += 1
        elif character in closing_to_opening:
            opening_character = closing_to_opening[character]
            nesting[opening_character] -= 1
            if nesting[opening_character] < 0:
                malformed_count += 1
                nesting[opening_character] = 0
        elif all(value == 0 for value in nesting.values()):
            match = key_pattern.match(masked, index)
            if match is not None:
                value_start = match.end()
                while value_start < len(masked) and masked[value_start].isspace():
                    value_start += 1
                targets: list[str] = []
                form = "malformed"
                value_stop = value_start
                if value_start < len(masked) and masked[value_start] == "{":
                    collection_depth = 0
                    value_stop = value_start
                    while value_stop < len(masked):
                        if masked[value_stop] == "{":
                            collection_depth += 1
                        elif masked[value_stop] == "}":
                            collection_depth -= 1
                            if collection_depth == 0:
                                break
                        value_stop += 1
                    if value_stop < len(masked):
                        form = "collection"
                        targets = [
                            item.group(0).removeprefix("%")
                            for item in symbol_pattern.finditer(
                                masked[value_start + 1 : value_stop]
                            )
                        ]
                    else:
                        malformed_count += 1
                elif value_start < len(masked) and masked[value_start] == "[":
                    value_stop = masked.find("]", value_start + 1)
                    if (
                        value_stop >= 0
                        and not masked[value_start + 1 : value_stop].strip()
                    ):
                        form = "empty"
                    else:
                        malformed_count += 1
                        value_stop = max(value_stop, value_start)
                else:
                    target = symbol_pattern.match(masked, value_start)
                    if target is not None:
                        form = "single"
                        targets = [target.group(0).removeprefix("%")]
                        value_stop = target.end()
                    else:
                        malformed_count += 1
                attributes.append(
                    {
                        "key": match.group("key"),
                        "form": form,
                        "targets": targets,
                    }
                )
                index = value_stop
        index += 1
    if any(nesting.values()):
        malformed_count += 1
    return {
        "attributes": attributes,
        "malformed_count": malformed_count,
        "passed": malformed_count == 0,
    }


def _matching_closing_brace(masked_text: str, opening: int) -> int | None:
    if opening < 0 or opening >= len(masked_text) or masked_text[opening] != "{":
        raise ValueError("entry-region opening must identify a brace")
    depth = 0
    for index in range(opening, len(masked_text)):
        character = masked_text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def _stablehlo_entry_call_ownership(text: str) -> dict[str, Any]:
    """Parse StableHLO with registered MLIR dialects and sanitize the graph."""
    try:
        from jax._src.interpreters import mlir
        from jaxlib.mlir import ir

        with mlir.make_ir_context() as context:
            module = ir.Module.parse(text, context)
            if module.operation.verify() is not True:
                raise RuntimeError("registered StableHLO module verification failed")
            functions = [
                operation
                for operation in module.body.operations
                if operation.operation.name == "func.func"
            ]
            symbols = {
                str(function.attributes["sym_name"]).strip('"'): function
                for function in functions
            }
            duplicate_computation_count = len(functions) - len(symbols)
            unexpected_top_level_operation_count = sum(
                operation.operation.name != "func.func"
                for operation in module.body.operations
            )
            entries = [
                name
                for name, function in symbols.items()
                if name == "main"
                and str(function.attributes.get("sym_visibility", '"public"')).strip(
                    '"'
                )
                == "public"
            ]
            calls: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []
            forbidden: list[dict[str, Any]] = []
            instruction_counts = dict.fromkeys(symbols, 0)
            call_index = 0

            def visit_block(
                block: Any,
                owner: str,
                region_depth: int,
                ancestors: tuple[str, ...],
            ) -> None:
                nonlocal call_index
                for operation in block.operations:
                    opcode = operation.operation.name
                    instruction_counts[owner] += 1
                    if opcode == "stablehlo.custom_call":
                        called_attribute = operation.attributes.get(
                            "called_computations"
                        )
                        called_text = (
                            str(called_attribute)
                            if called_attribute is not None
                            else "[]"
                        )
                        called_symbols = re.findall(
                            r"@([A-Za-z_][A-Za-z0-9_.$-]*)", called_text
                        )
                        called_is_empty = called_text.strip() in {
                            "[]",
                            "array<flat_symbol_ref>",
                        }
                        if called_attribute is not None and not called_is_empty:
                            forbidden.append(
                                {
                                    "opcode": "stablehlo.custom_call_called_computations",
                                    "owner_computation_sha256": _symbol_sha256(owner),
                                    "region_depth": region_depth,
                                }
                            )
                            if not called_symbols:
                                edges.append(
                                    {
                                        "caller": owner,
                                        "callee": "unresolved_called_computation",
                                        "region_depth": region_depth,
                                        "ancestor_opcodes": list(ancestors),
                                        "ordinary_direct_call": False,
                                        "factor": "unknown",
                                        "opcode": "stablehlo.custom_call",
                                    }
                                )
                            for callee in called_symbols:
                                edges.append(
                                    {
                                        "caller": owner,
                                        "callee": callee,
                                        "region_depth": region_depth,
                                        "ancestor_opcodes": list(ancestors),
                                        "ordinary_direct_call": False,
                                        "factor": "unknown",
                                        "opcode": "stablehlo.custom_call",
                                    }
                                )
                        calls.append(
                            {
                                "index": call_index,
                                "owner_computation_sha256": _symbol_sha256(owner),
                                "owner_is_entry": owner in entries,
                                "region_depth": region_depth,
                                "ancestor_opcodes": list(ancestors),
                                "direct_entry_owned": owner in entries
                                and region_depth == 0,
                                "has_nonempty_or_malformed_called_computations": (
                                    called_attribute is not None and not called_is_empty
                                ),
                            }
                        )
                        call_index += 1
                    elif opcode == "func.call":
                        callee = str(operation.attributes["callee"]).removeprefix("@")
                        edges.append(
                            {
                                "caller": owner,
                                "callee": callee,
                                "region_depth": region_depth,
                                "ancestor_opcodes": list(ancestors),
                                "ordinary_direct_call": region_depth == 0,
                                "factor": "one",
                                "opcode": "func.call",
                            }
                        )
                    elif opcode in {
                        "func.call_indirect",
                        "stablehlo.case",
                        "stablehlo.if",
                        "stablehlo.map",
                        "stablehlo.reduce",
                        "stablehlo.reduce_window",
                        "stablehlo.scatter",
                        "stablehlo.select_and_scatter",
                        "stablehlo.sort",
                        "stablehlo.while",
                    }:
                        forbidden.append(
                            {
                                "opcode": opcode,
                                "owner_computation_sha256": _symbol_sha256(owner),
                                "region_depth": region_depth,
                            }
                        )
                    for region in operation.regions:
                        for nested_block in region.blocks:
                            visit_block(
                                nested_block,
                                owner,
                                region_depth + 1,
                                (*ancestors, opcode),
                            )

            for symbol, function in symbols.items():
                for region in function.regions:
                    for block in region.blocks:
                        visit_block(block, symbol, 0, ())

            node_ids = {_symbol_sha256(symbol) for symbol in symbols}
            entry_id = _symbol_sha256(entries[0]) if len(entries) == 1 else None
            raw_edges = [
                (
                    _symbol_sha256(edge["caller"]),
                    _symbol_sha256(edge["callee"]),
                    edge["factor"],
                )
                for edge in edges
            ]
            graph = _saturated_entry_multiplicity(node_ids, entry_id, raw_edges)
            sanitized_edges = [
                {
                    "caller_computation_sha256": _symbol_sha256(edge["caller"]),
                    "callee_computation_sha256": _symbol_sha256(edge["callee"]),
                    "callee_resolved": edge["callee"] in symbols,
                    "region_depth": edge["region_depth"],
                    "ancestor_opcodes": edge["ancestor_opcodes"],
                    "ordinary_direct_call": edge["ordinary_direct_call"],
                    "factor": edge["factor"],
                    "opcode": edge["opcode"],
                }
                for edge in edges
            ]
            computations = [
                {
                    "computation_sha256": _symbol_sha256(symbol),
                    "is_entry": symbol in entries,
                    **graph["nodes"][_symbol_sha256(symbol)],
                    "custom_call_count": sum(
                        call["owner_computation_sha256"] == _symbol_sha256(symbol)
                        for call in calls
                    ),
                    "instruction_count": instruction_counts[symbol],
                }
                for symbol in sorted(symbols)
            ]
            graph_safe = (
                not graph["cycle_detected"]
                and graph["unknown_edge_count"] == 0
                and graph["unknown_factor_edge_count"] == 0
                and all(edge["ordinary_direct_call"] for edge in sanitized_edges)
            )
            checks = {
                "registered_mlir_parse_succeeded": True,
                "exactly_one_entry_computation": len(entries) == 1,
                "computation_symbols_are_unique": duplicate_computation_count == 0,
                "only_function_operations_are_top_level": (
                    unexpected_top_level_operation_count == 0
                ),
                "all_custom_calls_directly_owned_by_entry": len(calls) == 3
                and all(call["direct_entry_owned"] for call in calls),
                "no_callable_or_control_flow_container": not edges and not forbidden,
                "call_graph_is_acyclic_resolved_and_direct": graph_safe,
            }
            return {
                "parser": "registered_jax_mlir_context",
                "parse_succeeded": True,
                "entry_count": len(entries),
                "computation_count": len(computations),
                "custom_call_count": len(calls),
                "direct_entry_custom_call_count": sum(
                    call["direct_entry_owned"] for call in calls
                ),
                "forbidden_container_count": len(forbidden),
                "duplicate_computation_count": duplicate_computation_count,
                "unexpected_top_level_operation_count": (
                    unexpected_top_level_operation_count
                ),
                "calls": calls,
                "computations": computations,
                "call_edges": sanitized_edges,
                "forbidden_containers": forbidden,
                "entry_multiplicity": graph,
                "checks": checks,
                "passed": all(checks.values()),
                "raw_ir_emitted": False,
                "raw_symbols_emitted": False,
            }
    except Exception as error:
        checks = {
            "registered_mlir_parse_succeeded": False,
            "exactly_one_entry_computation": False,
            "computation_symbols_are_unique": False,
            "only_function_operations_are_top_level": False,
            "all_custom_calls_directly_owned_by_entry": False,
            "no_callable_or_control_flow_container": False,
            "call_graph_is_acyclic_resolved_and_direct": False,
        }
        return {
            "parser": "registered_jax_mlir_context",
            "parse_succeeded": False,
            "entry_count": 0,
            "computation_count": 0,
            "custom_call_count": 0,
            "direct_entry_custom_call_count": 0,
            "forbidden_container_count": 0,
            "duplicate_computation_count": 0,
            "unexpected_top_level_operation_count": 0,
            "calls": [],
            "computations": [],
            "call_edges": [],
            "forbidden_containers": [],
            "entry_multiplicity": {
                "cycle_detected": False,
                "unknown_edge_count": 0,
                "nodes": {},
            },
            "checks": checks,
            "passed": False,
            "parse_error_type": type(error).__name__,
            **_redacted_message_summary(error),
            "raw_ir_emitted": False,
            "raw_symbols_emitted": False,
        }


def _optimized_hlo_entry_call_ownership(text: str) -> dict[str, Any]:
    """Build a fail-closed sanitized graph from raw quote-masked HLO text."""
    quote_scan = _raw_ir_quote_scan(text)
    masked = str(quote_scan["masked"])
    raw_lines = text.splitlines()
    masked_lines = masked.splitlines()
    if len(raw_lines) != len(masked_lines):
        raise RuntimeError("optimized HLO quote masking changed line count")
    computation_symbol = r"%?[A-Za-z_][A-Za-z0-9_.-]*"
    instruction_symbol = r"%?(?:[A-Za-z_][A-Za-z0-9_.-]*|[0-9]+)"
    entry_header = re.compile(rf"^\s*ENTRY\s+(?P<name>{computation_symbol})\b")
    helper_header = re.compile(rf"^\s*(?P<name>{computation_symbol})(?:\s|\(|\{{)")
    assignment = re.compile(
        rf"^\s*(?:ROOT\s+)?(?P<name>{instruction_symbol})\s*=\s*(?P<definition>.*)$"
    )
    opcode_pattern = re.compile(r"(?P<opcode>[A-Za-z_][A-Za-z0-9_.-]*)\s*\(")
    forbidden_opcodes = {
        "async-start",
        "call-done",
        "call-start",
        "conditional",
        "fusion",
        "map",
        "reduce",
        "reduce-window",
        "scatter",
        "select-and-scatter",
        "sort",
        "while",
    }
    computations: dict[str, dict[str, Any]] = {}
    entries: list[str] = []
    calls: list[dict[str, Any]] = []
    call_owners: list[str] = []
    edges: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    unknown_callable = 0
    unknown_instruction_count = 0
    duplicate_computation_count = 0
    depth = 0
    owner: str | None = None
    call_index = 0
    balanced = quote_scan["passed"] is True

    def normalized_symbol(value: str) -> str:
        return value.removeprefix("%")

    for raw_line, masked_line in zip(raw_lines, masked_lines, strict=True):
        line_depth = depth
        stripped = masked_line.rstrip()
        if line_depth == 0 and stripped.endswith("{"):
            header = entry_header.match(masked_line)
            is_entry = header is not None
            if header is None:
                header = helper_header.match(masked_line)
            if header is not None and "=" not in masked_line:
                owner = normalized_symbol(header.group("name"))
                if owner in computations:
                    duplicate_computation_count += 1
                computations.setdefault(
                    owner,
                    {
                        "header_sha256": hashlib.sha256(
                            raw_line.encode("utf-8")
                        ).hexdigest(),
                        "is_entry": is_entry,
                        "custom_call_count": 0,
                        "instruction_count": 0,
                    },
                )
                if is_entry:
                    entries.append(owner)
        if owner is not None and line_depth >= 1:
            match = assignment.match(masked_line)
            if match is not None:
                computations[owner]["instruction_count"] += 1
                opcode_match = opcode_pattern.search(match.group("definition"))
                opcode = opcode_match.group("opcode") if opcode_match else None
                region_depth = line_depth - 1
                instruction_hash = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
                reference_summary = _top_level_hlo_computation_references(raw_line)
                reference_attributes = reference_summary["attributes"]
                reference_schema = [
                    {
                        "key": item["key"],
                        "form": item["form"],
                        "target_count": len(item["targets"]),
                    }
                    for item in reference_attributes
                ]
                referenced = [
                    target
                    for item in reference_attributes
                    for target in item["targets"]
                ]
                if opcode == "custom-call":
                    call_owners.append(owner)
                    calls.append(
                        {
                            "index": call_index,
                            "owner_computation_sha256": _symbol_sha256(owner),
                            "owner_is_entry": owner in entries,
                            "region_depth": region_depth,
                            "ancestor_opcodes": (
                                [] if region_depth == 0 else ["unknown_braced_region"]
                            ),
                            "direct_entry_owned": owner in entries
                            and region_depth == 0,
                            "instruction_sha256": instruction_hash,
                            "has_nonempty_or_malformed_called_computations": False,
                        }
                    )
                    computations[owner]["custom_call_count"] += 1
                    call_index += 1
                elif opcode == "call":
                    valid_call_reference = reference_summary[
                        "passed"
                    ] is True and reference_schema == [
                        {"key": "to_apply", "form": "single", "target_count": 1}
                    ]
                    if not valid_call_reference:
                        unknown_callable += 1
                    callees = referenced or ["unresolved_called_computation"]
                    for callee in callees:
                        edges.append(
                            {
                                "caller": owner,
                                "callee": callee,
                                "region_depth": region_depth,
                                "ancestor_opcodes": (
                                    []
                                    if region_depth == 0
                                    else ["unknown_braced_region"]
                                ),
                                "ordinary_direct_call": region_depth == 0,
                                "factor": "one" if valid_call_reference else "unknown",
                                "opcode": "call",
                                "instruction_sha256": instruction_hash,
                            }
                        )
                elif opcode in forbidden_opcodes:
                    callees = list(referenced)
                    if reference_summary["passed"] is not True and not callees:
                        callees.append("unresolved_called_computation")
                    for callee in callees:
                        edges.append(
                            {
                                "caller": owner,
                                "callee": callee,
                                "region_depth": region_depth,
                                "ancestor_opcodes": (
                                    []
                                    if region_depth == 0
                                    else ["unknown_braced_region"]
                                ),
                                "ordinary_direct_call": False,
                                "factor": "unknown",
                                "opcode": opcode,
                                "instruction_sha256": instruction_hash,
                            }
                        )
                    forbidden.append(
                        {
                            "opcode": opcode,
                            "owner_computation_sha256": _symbol_sha256(owner),
                            "region_depth": region_depth,
                            "ancestor_opcodes": (
                                [] if region_depth == 0 else ["unknown_braced_region"]
                            ),
                            "instruction_sha256": instruction_hash,
                            "reference_schema": reference_schema,
                            "reference_parse_passed": reference_summary["passed"],
                        }
                    )
                elif reference_attributes or reference_summary["passed"] is not True:
                    unknown_callable += 1
                    callees = list(referenced) or ["unresolved_called_computation"]
                    for callee in callees:
                        edges.append(
                            {
                                "caller": owner,
                                "callee": callee,
                                "region_depth": region_depth,
                                "ancestor_opcodes": (
                                    []
                                    if region_depth == 0
                                    else ["unknown_braced_region"]
                                ),
                                "ordinary_direct_call": False,
                                "factor": "unknown",
                                "opcode": "unknown_to_apply_instruction",
                                "instruction_sha256": instruction_hash,
                            }
                        )
                    forbidden.append(
                        {
                            "opcode": "unknown_to_apply_instruction",
                            "owner_computation_sha256": _symbol_sha256(owner),
                            "region_depth": region_depth,
                            "ancestor_opcodes": (
                                [] if region_depth == 0 else ["unknown_braced_region"]
                            ),
                            "instruction_sha256": instruction_hash,
                            "reference_schema": reference_schema,
                            "reference_parse_passed": reference_summary["passed"],
                        }
                    )
                elif opcode is None:
                    unknown_instruction_count += 1
        for character in masked_line:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth < 0:
                    balanced = False
        if owner is not None and depth == 0:
            owner = None
    if depth != 0:
        balanced = False

    custom_blocks = _raw_custom_call_blocks(text, "optimized_hlo")
    custom_call_block_count_matches_graph = len(custom_blocks) == len(calls)
    for index, (owner_name, block) in enumerate(
        zip(call_owners, custom_blocks, strict=False)
    ):
        reference_summary = _top_level_hlo_computation_references(block)
        attributes = reference_summary["attributes"]
        referenced = [target for item in attributes for target in item["targets"]]
        only_empty_called_computation_attributes = all(
            item["key"] in {"calls", "called_computations"}
            and item["form"] in {"empty", "collection"}
            and not item["targets"]
            for item in attributes
        )
        nonempty_or_malformed = (
            reference_summary["passed"] is not True
            or bool(referenced)
            or not only_empty_called_computation_attributes
        )
        calls[index]["has_nonempty_or_malformed_called_computations"] = (
            nonempty_or_malformed
        )
        if nonempty_or_malformed:
            forbidden.append(
                {
                    "opcode": "custom_call_called_computations",
                    "owner_computation_sha256": _symbol_sha256(owner_name),
                    "region_depth": calls[index]["region_depth"],
                    "ancestor_opcodes": calls[index]["ancestor_opcodes"],
                    "instruction_sha256": calls[index]["instruction_sha256"],
                }
            )
            if reference_summary["passed"] is not True and not referenced:
                referenced.append("unresolved_called_computation")
            for callee in referenced:
                edges.append(
                    {
                        "caller": owner_name,
                        "callee": callee,
                        "region_depth": calls[index]["region_depth"],
                        "ancestor_opcodes": calls[index]["ancestor_opcodes"],
                        "ordinary_direct_call": False,
                        "factor": "unknown",
                        "opcode": "custom-call",
                        "instruction_sha256": calls[index]["instruction_sha256"],
                    }
                )

    node_ids = {_symbol_sha256(name) for name in computations}
    entry_id = _symbol_sha256(entries[0]) if len(entries) == 1 else None
    raw_edges = [
        (
            _symbol_sha256(edge["caller"]),
            _symbol_sha256(edge["callee"]),
            edge["factor"],
        )
        for edge in edges
    ]
    graph = _saturated_entry_multiplicity(node_ids, entry_id, raw_edges)
    sanitized_edges = [
        {
            "caller_computation_sha256": _symbol_sha256(edge["caller"]),
            "callee_computation_sha256": _symbol_sha256(edge["callee"]),
            "callee_resolved": edge["callee"] in computations,
            "region_depth": edge["region_depth"],
            "ancestor_opcodes": edge["ancestor_opcodes"],
            "ordinary_direct_call": edge["ordinary_direct_call"],
            "factor": edge["factor"],
            "opcode": edge["opcode"],
            "instruction_sha256": edge["instruction_sha256"],
        }
        for edge in edges
    ]
    sanitized_computations = [
        {
            "computation_sha256": _symbol_sha256(name),
            "header_sha256": value["header_sha256"],
            "is_entry": name in entries,
            **graph["nodes"][_symbol_sha256(name)],
            "custom_call_count": value["custom_call_count"],
            "instruction_count": value["instruction_count"],
        }
        for name, value in sorted(computations.items())
    ]
    exactly_three_direct_entry_custom_calls = len(calls) == 3 and all(
        call["direct_entry_owned"] for call in calls
    )
    nonentry_computations_have_zero_custom_calls = all(
        item["is_entry"] or item["custom_call_count"] == 0
        for item in sanitized_computations
    )
    custom_calls_have_no_called_computations = all(
        not call["has_nonempty_or_malformed_called_computations"] for call in calls
    )
    entry_computation_sha256 = entry_id if len(entries) == 1 else None
    fusion_edges_are_independent_entry_helpers = all(
        edge["opcode"] == "fusion"
        and edge["caller_computation_sha256"] == entry_computation_sha256
        and edge["callee_resolved"]
        and edge["callee_computation_sha256"] != entry_computation_sha256
        and edge["region_depth"] == 0
        and not edge["ancestor_opcodes"]
        and edge["factor"] == "unknown"
        for edge in sanitized_edges
    )
    fusion_containers_are_direct_entry_helpers = all(
        item["opcode"] == "fusion"
        and item["owner_computation_sha256"] == entry_computation_sha256
        and item["region_depth"] == 0
        and not item["ancestor_opcodes"]
        and item.get("reference_parse_passed") is True
        and item.get("reference_schema")
        == [{"key": "calls", "form": "single", "target_count": 1}]
        for item in forbidden
    )
    fusion_sites_are_one_to_one = len(sanitized_edges) == len(forbidden) and {
        edge["instruction_sha256"] for edge in sanitized_edges
    } == {item["instruction_sha256"] for item in forbidden}
    independent_fusion_helpers_safe = (
        fusion_edges_are_independent_entry_helpers
        and fusion_containers_are_direct_entry_helpers
        and fusion_sites_are_one_to_one
    )
    graph_is_custom_call_independent = (
        not graph["cycle_detected"]
        and graph["unknown_edge_count"] == 0
        and graph["unknown_factor_edge_count"] == len(sanitized_edges)
        and unknown_callable == 0
        and duplicate_computation_count == 0
        and unknown_instruction_count == 0
        and custom_call_block_count_matches_graph
        and exactly_three_direct_entry_custom_calls
        and sum(item["custom_call_count"] for item in sanitized_computations) == 3
        and nonentry_computations_have_zero_custom_calls
        and custom_calls_have_no_called_computations
        and independent_fusion_helpers_safe
    )
    checks = {
        "raw_quote_aware_parse_balanced": balanced,
        "raw_quote_escapes_are_well_formed": quote_scan["passed"] is True,
        "exactly_one_entry_computation": len(entries) == 1,
        "computation_symbols_are_unique": duplicate_computation_count == 0,
        "every_instruction_opcode_was_parsed": unknown_instruction_count == 0,
        "custom_call_blocks_match_instruction_graph": (
            custom_call_block_count_matches_graph
        ),
        "exactly_three_custom_calls_all_directly_owned_by_entry": (
            exactly_three_direct_entry_custom_calls
        ),
        "all_nonentry_computations_have_zero_custom_calls": (
            nonentry_computations_have_zero_custom_calls
        ),
        "custom_calls_have_no_called_computations": (
            custom_calls_have_no_called_computations
        ),
        "only_independent_direct_entry_fusion_helpers_are_present": (
            independent_fusion_helpers_safe
        ),
        "no_container_can_own_or_duplicate_a_custom_call": (
            graph_is_custom_call_independent
        ),
        "call_graph_is_acyclic_resolved_and_custom_call_independent": (
            graph_is_custom_call_independent
        ),
    }
    return {
        "parser": "raw_quote_aware_hlo_computation_graph",
        "parse_succeeded": balanced,
        "entry_count": len(entries),
        "computation_count": len(sanitized_computations),
        "custom_call_count": len(calls),
        "direct_entry_custom_call_count": sum(
            call["direct_entry_owned"] for call in calls
        ),
        "forbidden_container_count": len(forbidden) + unknown_callable,
        "calls": calls,
        "computations": sanitized_computations,
        "call_edges": sanitized_edges,
        "forbidden_containers": forbidden,
        "unknown_callable_count": unknown_callable,
        "unknown_instruction_count": unknown_instruction_count,
        "duplicate_computation_count": duplicate_computation_count,
        "custom_call_block_count_matches_graph": (
            custom_call_block_count_matches_graph
        ),
        "allowed_independent_entry_fusion_helper_count": (
            len(sanitized_edges) if independent_fusion_helpers_safe else 0
        ),
        "entry_multiplicity": graph,
        "raw_quote_scan": {
            key: value for key, value in quote_scan.items() if key != "masked"
        },
        "checks": checks,
        "passed": all(checks.values()),
        "raw_ir_emitted": False,
        "raw_symbols_emitted": False,
    }


def _entry_call_ownership(text: str, dialect: str) -> dict[str, Any]:
    if dialect == "stablehlo":
        return _stablehlo_entry_call_ownership(text)
    if dialect == "optimized_hlo":
        return _optimized_hlo_entry_call_ownership(text)
    raise ValueError("unsupported VJP IR dialect")


def _is_bounded_marker_like(token: str) -> bool:
    normalized = re.sub(r"[-.$]+", "_", token.lower())
    return any(
        stem in normalized
        for stem in (
            "query_bounded_gqa_forward_q",
            "query_bounded_gqa_dq_q",
            "query_bounded_gqa_dkdv_q",
        )
    )


def _strict_vjp_ir_summary(text: str, dialect: str) -> dict[str, Any]:
    compile_probe = _compile_probe()
    definitions = compile_probe._metadata_definitions(text)
    raw_blocks = _raw_custom_call_blocks(text, dialect)
    blocks = [
        compile_probe._resolved_block_metadata(block, definitions)
        for block in raw_blocks
    ]
    decoded_text = _decode_ir(text)
    ownership = _entry_call_ownership(text, dialect)
    all_tokens = _IR_NAME_TOKEN_PATTERN.findall(decoded_text)
    bounded_tokens = [token for token in all_tokens if _is_bounded_marker_like(token)]
    unexpected = [
        token for token in bounded_tokens if token not in _EXPECTED_MARKERS.values()
    ]
    masked_text = _mask_ir_quoted_content(text)
    if dialect == "stablehlo":
        textual_count = len(re.findall(r"\bstablehlo\.custom_call\b", masked_text))
        while_count = len(re.findall(r"\bstablehlo\.while\b", masked_text))
    elif dialect == "optimized_hlo":
        textual_count = len(re.findall(r"\bcustom-call\(", masked_text))
        while_count = len(re.findall(r"\bwhile\s*\(", masked_text))
    else:
        raise ValueError("unsupported VJP IR dialect")
    decoded_masked = _mask_ir_quoted_content(decoded_text)
    decoded_masked_call_count = (
        len(re.findall(r"\bstablehlo\.custom_call\b", decoded_masked))
        if dialect == "stablehlo"
        else len(re.findall(r"\bcustom-call\(", decoded_masked))
    )
    raw_quote_scan = _raw_ir_quote_scan(text)

    calls: list[dict[str, Any]] = []
    marker_counts = dict.fromkeys(_EXPECTED_MARKERS, 0)
    all_call_checks: list[bool] = []
    for index, block in enumerate(blocks):
        target_occurrences = _isolated_custom_call_target_occurrences(block, dialect)
        tokens = _IR_NAME_TOKEN_PATTERN.findall(_decode_ir(block))
        kinds = [kind for kind, marker in _EXPECTED_MARKERS.items() if marker in tokens]
        for kind in kinds:
            marker_counts[kind] += 1
        query_start = _nonzero_probe()._canonical_raw_metadata_field(
            _decode_ir(block), "query_start", 0
        )
        query_size = _nonzero_probe()._canonical_raw_metadata_field(
            _decode_ir(block), "query_size", _SEQUENCE_LENGTH
        )
        checks = {
            "sole_exact_target": target_occurrences == [_EXACT_TARGET],
            "exactly_one_expected_marker": len(kinds) == 1,
            "query_start_is_exact_canonical_zero": query_start["passed"],
            "query_size_is_exact_canonical_512": query_size["passed"],
        }
        all_call_checks.append(all(checks.values()))
        calls.append(
            {
                "index": index,
                "kind": kinds[0] if len(kinds) == 1 else None,
                "target_count": len(target_occurrences),
                "sole_target_matches_expected": target_occurrences == [_EXACT_TARGET],
                "sole_target_sha256": (
                    hashlib.sha256(target_occurrences[0].encode("utf-8")).hexdigest()
                    if len(target_occurrences) == 1
                    else None
                ),
                "query_start": query_start,
                "query_size": query_size,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    checks = {
        "parser_matches_textual_custom_call_count": len(raw_blocks) == textual_count,
        "exactly_three_custom_calls_total": len(raw_blocks) == textual_count == 3,
        "all_three_calls_pass_target_marker_and_metadata": len(all_call_checks) == 3
        and all(all_call_checks),
        "each_expected_marker_occurs_in_exactly_one_call": all(
            marker_counts[kind] == 1 for kind in _EXPECTED_MARKERS
        ),
        "no_unexpected_or_lookalike_bounded_markers": not unexpected,
        "all_calls_direct_entry_and_container_independent": (ownership["passed"]),
        "no_outer_while": while_count == 0,
    }
    return {
        "dialect": dialect,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "custom_call_count": len(raw_blocks),
        "custom_call_token_count": textual_count,
        "marker_call_counts": marker_counts,
        "unexpected_bounded_marker_occurrences": len(unexpected),
        "while_count": while_count,
        "calls": calls,
        "entry_call_ownership": ownership,
        "quote_lexing_diagnostic": {
            "syntax_was_lexed_before_local_escape_decoding": True,
            "raw_unquoted_custom_call_count": textual_count,
            "decoded_then_masked_custom_call_count_diagnostic_only": (
                decoded_masked_call_count
            ),
            "raw_mlir_hex_quote_escape_count": len(
                re.findall(r"\\22", text, flags=re.IGNORECASE)
            ),
            "raw_backslash_quote_escape_count": len(re.findall(r'\\"', text)),
            "raw_quote_delimiter_count": raw_quote_scan["quote_delimiter_count"],
            "raw_escape_count": raw_quote_scan["escape_count"],
            "raw_invalid_escape_count": raw_quote_scan["invalid_escape_count"],
            "raw_unterminated_quote": raw_quote_scan["unterminated_quote"],
            "raw_quote_scan_passed": raw_quote_scan["passed"],
            "raw_payload_emitted": False,
        },
        "checks": checks,
        "raw_ir_emitted": False,
        "passed": all(checks.values()),
    }


def _structural_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    dialects = [str(summary.get("dialect")) for summary in summaries]
    exact = len(summaries) == 2 and sorted(dialects) == [
        "optimized_hlo",
        "stablehlo",
    ]
    per = {str(summary.get("dialect")): summary for summary in summaries}
    checks = {
        "exactly_one_summary_per_required_dialect": exact,
        "stablehlo_passed": exact and per.get("stablehlo", {}).get("passed") is True,
        "optimized_hlo_passed": exact
        and per.get("optimized_hlo", {}).get("passed") is True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _compiled_memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    values = {
        name: memory.get(name)
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
        )
    }
    available = memory.get("available") is True and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values.values()
    )
    checks = {
        "memory_analysis_available": available,
        "argument_bytes_exact": values["argument_size_in_bytes"]
        == _EXPECTED_ARGUMENT_BYTES,
        "output_bytes_exact": values["output_size_in_bytes"] == _EXPECTED_OUTPUT_BYTES,
        "alias_bytes_exact": values["alias_size_in_bytes"] == _EXPECTED_ALIAS_BYTES,
        "temporary_bytes_at_most_64_mib": available
        and values["temp_size_in_bytes"] <= _MAX_TEMP_BYTES,
    }
    return {
        "expected_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
        "expected_output_bytes": _EXPECTED_OUTPUT_BYTES,
        "expected_alias_bytes": _EXPECTED_ALIAS_BYTES,
        "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        "checks": checks,
        "passed": all(checks.values()),
    }


class _CheckedVjpExecutable:
    __slots__ = ("_compiled", "_consumed", "_counters", "proof")

    def __init__(
        self,
        compiled: Any,
        *,
        proof: dict[str, Any],
        counters: dict[str, int],
        token: object,
    ) -> None:
        if token is not _CHECKED_CAPABILITY_TOKEN or proof.get("passed") is not True:
            raise RuntimeError("refusing to expose VJP executable without passed gates")
        self._compiled = compiled
        self._consumed = False
        self._counters = counters
        self.proof = proof

    def invoke(
        self, jax: Any, arguments: tuple[Any, ...], on_started: Callable[[], None]
    ) -> Any:
        if self._consumed:
            raise RuntimeError("checked VJP capability was already consumed")
        self._consumed = True
        self._counters["candidate_attempts"] += 1
        self._counters["checked_executable_invocations"] += 1
        on_started()
        result = self._compiled(*arguments)
        result = jax.block_until_ready(result)
        self._counters["candidate_completions"] += 1
        return result


def _wrap_checked(
    compiled: Any, proof: dict[str, Any], counters: dict[str, int]
) -> _CheckedVjpExecutable:
    return _CheckedVjpExecutable(
        compiled,
        proof=proof,
        counters=counters,
        token=_CHECKED_CAPABILITY_TOKEN,
    )


def _shape_signature(jax: Any, jnp: Any) -> tuple[Any, ...]:
    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    return (
        jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct((_BATCH_SIZE, _SEQUENCE_LENGTH), jnp.int32),
        jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
    )


def _compile_vjp_artifact(
    jax: Any,
    jnp: Any,
    query_bounded_gqa: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, dict[str, Any]]:
    def forward_and_vjp(
        q_arg: Any, k_arg: Any, v_arg: Any, mask_arg: Any, dout_arg: Any
    ) -> tuple[Any, Any, Any, Any]:
        value, pullback = jax.vjp(
            lambda q_item, k_item, v_item: query_bounded_gqa(
                q_item,
                k_item,
                v_item,
                mask_arg,
                scale=_ATTENTION_SCALE,
                query_chunk_size=_QUERY_CHUNK_SIZE,
                block_q=_BLOCK_Q,
                block_k=_BLOCK_K,
                backward_block_q=_BACKWARD_BLOCK_Q,
                backward_block_k=_BACKWARD_BLOCK_K,
                interpret=False,
            ),
            q_arg,
            k_arg,
            v_arg,
        )
        dq, dk, dv = pullback(dout_arg)
        return value, dq, dk, dv

    _emit(
        {
            "record_type": "stage",
            "stage": "vjp_lower_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["lower_attempts"] += 1
    lower_start = time.perf_counter()
    try:
        lowered = jax.jit(forward_and_vjp).lower(*_shape_signature(jax, jnp))
        lower_seconds = time.perf_counter() - lower_start
        counters["lower_completions"] += 1
        stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = _strict_vjp_ir_summary(stablehlo_text, "stablehlo")
        del stablehlo_text
        _emit(
            {
                "record_type": "lowered",
                "stage": "vjp_lower_complete_metadata_only",
                "timestamp": _utc_now(),
                "lower_seconds": _json_safe_duration(lower_seconds),
                "stablehlo": stablehlo,
                "raw_ir_emitted": False,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_vjp_lower_attempt", counters
        )

    _emit(
        {
            "record_type": "stage",
            "stage": "vjp_compile_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["compile_attempts"] += 1
    compiled = None
    retain_compiled_artifact = False
    compile_start = time.perf_counter()
    try:
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - compile_start
        counters["compile_completions"] += 1
        optimized_hlo_text = compiled.as_text()
        optimized_hlo = _strict_vjp_ir_summary(optimized_hlo_text, "optimized_hlo")
        del optimized_hlo_text
        memory = _compile_probe()._compiled_memory(compiled)
        structural = _structural_gate(stablehlo, optimized_hlo)
        memory_gate = _compiled_memory_gate(memory)
        proof = {
            "structural_gate_passed": structural["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "exact_logical_dispatches": dict.fromkeys(_EXPECTED_MARKERS, 1),
            "explicit_scale_exact_fraction": "3/32",
            "passed": structural["passed"] and memory_gate["passed"],
        }
        report = {
            "record_type": "vjp_compiled",
            "stage": "vjp_compile_release_gate",
            "timestamp": _utc_now(),
            "lower_seconds": _json_safe_duration(lower_seconds),
            "compile_seconds": _json_safe_duration(compile_seconds),
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "compiled_memory": memory,
            "structural_gate": structural,
            "compiled_memory_gate": memory_gate,
            "release_gate": proof,
            "raw_ir_emitted": False,
            "counters": dict(counters),
        }
        _emit(report, output)
        retain_compiled_artifact = True
    finally:
        if compiled is not None and not retain_compiled_artifact:
            del compiled
        _journal_checkpoint(
            require_clean_boot, output, "after_vjp_compile_attempt", counters
        )
    del lowered
    return compiled, report


def _release_checked_vjp(
    compiled: Any, report: dict[str, Any], counters: dict[str, int]
) -> _CheckedVjpExecutable:
    counters["checked_capability_release_attempts"] += 1
    proof = report.get("release_gate")
    if not isinstance(proof, dict) or proof.get("passed") is not True:
        raise RuntimeError("VJP executable failed structural or memory release gate")
    checked = _wrap_checked(compiled, proof, counters)
    counters["checked_capability_release_completions"] += 1
    return checked


def _run_compile_diagnostic(
    jax: Any,
    jnp: Any,
    query_bounded_gqa: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> int:
    compiled = None
    try:
        compiled, report = _compile_vjp_artifact(
            jax, jnp, query_bounded_gqa, require_clean_boot, counters, output
        )
    finally:
        if compiled is not None:
            del compiled
    expected = _compile_diagnostic_completed_counters()
    if counters != expected:
        raise RuntimeError("compile diagnostic counter contract was not exact")
    _emit(
        {
            "record_type": "compile_diagnostic_completed",
            "timestamp": _utc_now(),
            "status": (
                "structure_and_memory_passed_no_release"
                if report["release_gate"]["passed"]
                else "structure_or_memory_failed_no_release"
            ),
            "structural_gate": report["structural_gate"],
            "compiled_memory_gate": report["compiled_memory_gate"],
            "release_gate_observed_but_not_authorized": report["release_gate"],
            "checked_capability_created_or_released": False,
            "host_reference_constructed": False,
            "host_or_device_inputs_constructed": False,
            "executable_invoked": False,
            "device_output_retrieved": False,
            "raw_ir_emitted": False,
            "counters": dict(counters),
        },
        output,
    )
    return 0


def _iid_nonzero_grid(
    np: Any,
    ml_dtypes: Any,
    rng: Any,
    shape: tuple[int, ...],
    maximum_magnitude: int,
) -> Any:
    magnitudes = rng.integers(1, maximum_magnitude + 1, size=shape, dtype=np.int16)
    signs = 2 * rng.integers(0, 2, size=shape, dtype=np.int16) - 1
    result = (
        (magnitudes * signs).astype(np.float32) / np.float32(_GRID_DENOMINATOR)
    ).astype(ml_dtypes.bfloat16)
    if tuple(result.shape) != shape or int(np.count_nonzero(result)) != result.size:
        raise RuntimeError("nonzero BF16 grid construction violated its contract")
    return result


def _validate_oracle_inputs(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    key_mask: Any,
    dout: Any,
    *,
    require_loss_masked_padding: bool = True,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or dout.ndim != 4:
        raise RuntimeError("VJP oracle requires rank-four Q/K/V/dout")
    batch, sequence, query_heads, head_dim = q.shape
    if (
        batch != 1
        or k.shape[0] != batch
        or v.shape != k.shape
        or k.shape[1] != sequence
        or k.shape[3] != head_dim
        or dout.shape != q.shape
        or query_heads % k.shape[2]
        or sequence <= 0
        or head_dim <= 0
    ):
        raise RuntimeError("VJP oracle arrays violate grouped self-attention shapes")
    if (
        not isinstance(key_mask, np.ndarray)
        or key_mask.shape != (batch, sequence)
        or key_mask.dtype != np.dtype(np.int32)
        or not bool(np.all((key_mask == 0) | (key_mask == 1)))
    ):
        raise RuntimeError("VJP oracle requires an exact binary int32 key mask")
    valid_tokens = int(np.sum(key_mask[0], dtype=np.int64))
    expected_mask = np.zeros((sequence,), dtype=np.int32)
    expected_mask[:valid_tokens] = 1
    if valid_tokens <= 0 or not bool(np.array_equal(key_mask[0], expected_mask)):
        raise RuntimeError("VJP oracle requires a nonempty right-padded key mask")
    if require_loss_masked_padding and valid_tokens < sequence:
        padded_dout = np.asarray(dout[:, valid_tokens:])
        if any(padded_dout.tobytes(order="C")):
            raise RuntimeError(
                "padded VJP oracle requires bitwise-positive-zero dout padding rows"
            )
    return sequence, query_heads, k.shape[2], head_dim, valid_tokens


def _tiled_causal_gqa_forward_vjp_oracle(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    key_mask: Any,
    dout: Any,
    *,
    scale: float,
    query_tile: int = _ORACLE_QUERY_TILE,
    key_tile: int = _ORACLE_KEY_TILE,
    _require_loss_masked_padding: bool = True,
) -> tuple[tuple[Any, Any, Any, Any], dict[str, Any]]:
    sequence, query_heads, kv_heads, head_dim, valid_tokens = _validate_oracle_inputs(
        np,
        q,
        k,
        v,
        key_mask,
        dout,
        require_loss_masked_padding=_require_loss_masked_padding,
    )
    if query_tile <= 0 or key_tile <= 0 or not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("VJP oracle tile sizes and scale must be positive")
    group_size = query_heads // kv_heads
    output = np.empty(q.shape, dtype=np.float32)
    dq = np.zeros(q.shape, dtype=np.float32)
    dk = np.zeros(k.shape, dtype=np.float32)
    dv = np.zeros(v.shape, dtype=np.float32)
    maximum_absolute_valid_logit = 0.0

    for query_head in range(query_heads):
        kv_head = query_head // group_size
        for query_offset in range(0, sequence, query_tile):
            query_stop = min(query_offset + query_tile, sequence)
            rows = query_stop - query_offset
            query_positions = np.arange(query_offset, query_stop, dtype=np.int32)
            q_block = np.asarray(q[0, query_offset:query_stop, query_head], np.float32)
            dout_block = np.asarray(
                dout[0, query_offset:query_stop, query_head], np.float32
            )

            row_max = np.full((rows,), -np.inf, dtype=np.float32)
            row_sum = np.zeros((rows,), dtype=np.float32)
            for key_offset in range(0, query_stop, key_tile):
                key_stop = min(key_offset + key_tile, query_stop)
                key_positions = np.arange(key_offset, key_stop, dtype=np.int32)
                k_block = np.asarray(k[0, key_offset:key_stop, kv_head], np.float32)
                logits = (q_block @ k_block.T).astype(np.float32, copy=False)
                logits *= np.float32(scale)
                key_valid = key_mask[0, key_offset:key_stop].astype(
                    np.bool_, copy=False
                )
                valid = (key_positions[None, :] <= query_positions[:, None]) & (
                    key_valid[None, :]
                )
                valid_logits = logits[valid]
                if valid_logits.size:
                    maximum_absolute_valid_logit = max(
                        maximum_absolute_valid_logit,
                        float(np.max(np.abs(valid_logits))),
                    )
                logits = np.where(valid, logits, np.float32(-np.inf))
                block_max = np.max(logits, axis=1)
                next_max = np.maximum(row_max, block_max)
                correction = np.exp(row_max - next_max).astype(np.float32, copy=False)
                probabilities = np.exp(logits - next_max[:, None]).astype(
                    np.float32, copy=False
                )
                row_sum = correction * row_sum + np.sum(
                    probabilities, axis=1, dtype=np.float32
                )
                row_max = next_max

            output_block = np.zeros((rows, head_dim), dtype=np.float32)
            for key_offset in range(0, query_stop, key_tile):
                key_stop = min(key_offset + key_tile, query_stop)
                key_positions = np.arange(key_offset, key_stop, dtype=np.int32)
                k_block = np.asarray(k[0, key_offset:key_stop, kv_head], np.float32)
                v_block = np.asarray(v[0, key_offset:key_stop, kv_head], np.float32)
                logits = (q_block @ k_block.T).astype(np.float32, copy=False)
                logits *= np.float32(scale)
                key_valid = key_mask[0, key_offset:key_stop].astype(
                    np.bool_, copy=False
                )
                valid = (key_positions[None, :] <= query_positions[:, None]) & (
                    key_valid[None, :]
                )
                probabilities = np.exp(
                    np.where(valid, logits - row_max[:, None], np.float32(-np.inf))
                ).astype(np.float32, copy=False)
                probabilities /= row_sum[:, None]
                output_block += (probabilities @ v_block).astype(np.float32, copy=False)
                dv[0, key_offset:key_stop, kv_head] += (
                    probabilities.T @ dout_block
                ).astype(np.float32, copy=False)
            output[0, query_offset:query_stop, query_head] = output_block
            delta = np.sum(output_block * dout_block, axis=1, dtype=np.float32)

            dq_block = np.zeros((rows, head_dim), dtype=np.float32)
            for key_offset in range(0, query_stop, key_tile):
                key_stop = min(key_offset + key_tile, query_stop)
                key_positions = np.arange(key_offset, key_stop, dtype=np.int32)
                k_block = np.asarray(k[0, key_offset:key_stop, kv_head], np.float32)
                v_block = np.asarray(v[0, key_offset:key_stop, kv_head], np.float32)
                logits = (q_block @ k_block.T).astype(np.float32, copy=False)
                logits *= np.float32(scale)
                key_valid = key_mask[0, key_offset:key_stop].astype(
                    np.bool_, copy=False
                )
                valid = (key_positions[None, :] <= query_positions[:, None]) & (
                    key_valid[None, :]
                )
                probabilities = np.exp(
                    np.where(valid, logits - row_max[:, None], np.float32(-np.inf))
                ).astype(np.float32, copy=False)
                probabilities /= row_sum[:, None]
                dprobabilities = (dout_block @ v_block.T).astype(np.float32, copy=False)
                dscores = probabilities * (dprobabilities - delta[:, None])
                dscores *= np.float32(scale)
                dq_block += (dscores @ k_block).astype(np.float32, copy=False)
                dk[0, key_offset:key_stop, kv_head] += (dscores.T @ q_block).astype(
                    np.float32, copy=False
                )
            dq[0, query_offset:query_stop, query_head] = dq_block

    if not all(bool(np.all(np.isfinite(item))) for item in (output, dq, dk, dv)):
        raise RuntimeError("tiled FP32 VJP oracle produced non-finite tensors")
    conservative_scratch_bytes = (
        4
        * (
            6 * query_tile * key_tile
            + 5 * query_tile * head_dim
            + 4 * key_tile * head_dim
            + 12 * query_tile
            + 4 * key_tile
        )
        + query_tile * key_tile
        + 4 * (2 * query_tile + 2 * key_tile)
    )
    return (output, dq, dk, dv), {
        "implementation": "independent_host_fp32_three_pass_tiled_causal_gqa_forward_vjp",
        "query_tile": query_tile,
        "key_tile": key_tile,
        "full_t_by_t_matrix_constructed": False,
        "q_k_v_dout_converted_to_fp32_by_active_tile_only": True,
        "global_fp32_outputs": ["output", "dq", "dk", "dv"],
        "conservative_accounted_numpy_array_scratch_bytes": conservative_scratch_bytes,
        "scratch_accounting_excludes": [
            "required_input_output_and_gradient_arrays",
            "python_and_numpy_object_overhead",
            "numpy_blas_internal_workspace",
            "allocator_fragmentation",
        ],
        "scratch_bytes_are_conservative_accounting_not_measured_peak": True,
        "observed_maximum_absolute_valid_logit": maximum_absolute_valid_logit,
        "scale": float(scale),
        "valid_tokens": valid_tokens,
        "right_padded_key_mask": valid_tokens < sequence,
        "loss_masked_dout_padding_required": _require_loss_masked_padding,
        "accelerator_used": False,
    }


def _dense_causal_gqa_forward_vjp_reference(
    np: Any, q: Any, k: Any, v: Any, key_mask: Any, dout: Any, *, scale: float
) -> tuple[Any, Any, Any, Any]:
    sequence, query_heads, kv_heads, head_dim, valid_tokens = _validate_oracle_inputs(
        np, q, k, v, key_mask, dout
    )
    if sequence > _SEQUENCE_LENGTH:
        raise RuntimeError("dense reference is restricted to bounded CPU tests")
    group_size = query_heads // kv_heads
    q_fp32 = np.asarray(q, np.float32)[0]
    k_fp32 = np.asarray(k, np.float32)[0]
    v_fp32 = np.asarray(v, np.float32)[0]
    dout_fp32 = np.asarray(dout, np.float32)[0]
    output = np.empty(q.shape, np.float32)
    dq = np.zeros(q.shape, np.float32)
    dk = np.zeros(k.shape, np.float32)
    dv = np.zeros(v.shape, np.float32)
    invalid = np.triu(np.ones((sequence, sequence), dtype=np.bool_), k=1)
    invalid[:, valid_tokens:] = True
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        logits = (q_fp32[:, query_head] @ k_fp32[:, kv_head].T).astype(
            np.float32, copy=False
        )
        logits *= np.float32(scale)
        logits[invalid] = -np.inf
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits).astype(np.float32, copy=False)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True, dtype=np.float32)
        output_head = (probabilities @ v_fp32[:, kv_head]).astype(
            np.float32, copy=False
        )
        output[0, :, query_head] = output_head
        delta = np.sum(output_head * dout_fp32[:, query_head], axis=1, dtype=np.float32)
        dprobabilities = (dout_fp32[:, query_head] @ v_fp32[:, kv_head].T).astype(
            np.float32, copy=False
        )
        dscores = probabilities * (dprobabilities - delta[:, None])
        dscores *= np.float32(scale)
        dq[0, :, query_head] = dscores @ k_fp32[:, kv_head]
        dk[0, :, kv_head] += dscores.T @ q_fp32[:, query_head]
        dv[0, :, kv_head] += probabilities.T @ dout_fp32[:, query_head]
    return output, dq, dk, dv


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return _nonzero_probe()._array_manifest(name, value)


def _float_difference_metrics(np: Any, actual: Any, expected: Any) -> dict[str, float]:
    actual64 = np.asarray(actual, np.float64).ravel()
    expected64 = np.asarray(expected, np.float64).ravel()
    difference = actual64 - expected64
    actual_norm = float(np.linalg.norm(actual64))
    expected_norm = float(np.linalg.norm(expected64))
    denominator = max(expected_norm, float(np.finfo(np.float64).tiny))
    cosine_denominator = max(
        actual_norm * expected_norm, float(np.finfo(np.float64).tiny)
    )
    return {
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "cosine": float(np.vdot(actual64, expected64) / cosine_denominator),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _bitwise_positive_zero(np: Any, value: Any) -> bool:
    return not any(np.asarray(value).tobytes(order="C"))


def _valid385_sensitivity_controls(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    key_mask: Any,
    masked_dout: Any,
    unmasked_dout: Any,
    expected: tuple[Any, ...],
) -> dict[str, Any]:
    wrong_key_mask = np.ones_like(key_mask, dtype=np.int32)
    ignored_key_mask, _ = _tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        wrong_key_mask,
        masked_dout,
        scale=_ATTENTION_SCALE,
    )
    ignored_loss_mask, _ = _tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        key_mask,
        unmasked_dout,
        scale=_ATTENTION_SCALE,
        _require_loss_masked_padding=False,
    )
    valid_tokens = _CASE_VALID_TOKENS[_VALID385_CASE]
    wrong_valid384_mask = np.zeros_like(key_mask, dtype=np.int32)
    wrong_valid384_mask[:, : valid_tokens - 1] = 1
    wrong_valid384, _ = _tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        wrong_valid384_mask,
        masked_dout,
        scale=_ATTENTION_SCALE,
        _require_loss_masked_padding=False,
    )
    wrong_valid386_mask = np.zeros_like(key_mask, dtype=np.int32)
    wrong_valid386_mask[:, : valid_tokens + 1] = 1
    wrong_valid386, _ = _tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        wrong_valid386_mask,
        masked_dout,
        scale=_ATTENTION_SCALE,
    )
    expected_output, _expected_dq, _expected_dk, _expected_dv = expected
    affected_output_metrics = _float_difference_metrics(
        np,
        ignored_key_mask[0][:, valid_tokens:],
        expected_output[:, valid_tokens:],
    )
    boundary_output_metrics = {
        "384_wrong_valid384": _float_difference_metrics(
            np,
            wrong_valid384[0][:, valid_tokens - 1 : valid_tokens],
            expected_output[:, valid_tokens - 1 : valid_tokens],
        ),
        "385_wrong_valid386": _float_difference_metrics(
            np,
            wrong_valid386[0][:, valid_tokens : valid_tokens + 1],
            expected_output[:, valid_tokens : valid_tokens + 1],
        ),
        "511_ignored_key_mask": _float_difference_metrics(
            np,
            ignored_key_mask[0][:, 511:512],
            expected_output[:, 511:512],
        ),
    }
    ignored_loss_gradient_metrics = {
        name: _float_difference_metrics(np, wrong, correct)
        for name, wrong, correct in zip(
            ("dq", "dk", "dv"),
            ignored_loss_mask[1:],
            expected[1:],
            strict=True,
        )
    }
    hashes = {
        "ignored_key_mask_output": _array_manifest(
            "ignored_key_mask_output", ignored_key_mask[0]
        )["sha256"],
        "wrong_valid384_output": _array_manifest(
            "wrong_valid384_output", wrong_valid384[0]
        )["sha256"],
        "wrong_valid386_output": _array_manifest(
            "wrong_valid386_output", wrong_valid386[0]
        )["sha256"],
        **{
            f"ignored_loss_mask_{name}": _array_manifest(
                f"ignored_loss_mask_{name}", value
            )["sha256"]
            for name, value in zip(
                ("dq", "dk", "dv"), ignored_loss_mask[1:], strict=True
            )
        },
    }
    affected_key_mask_decisive = (
        affected_output_metrics["relative_l2"] >= _MAX_RELATIVE_L2
        or affected_output_metrics["cosine"] < _MIN_COSINE
        or affected_output_metrics["max_abs"] > _MAX_ABSOLUTE_ERROR
    )
    boundary_key_mask_decisive = {
        name: (
            item["relative_l2"] >= _MAX_RELATIVE_L2
            or item["cosine"] < _MIN_COSINE
            or item["max_abs"] > _MAX_ABSOLUTE_ERROR
        )
        for name, item in boundary_output_metrics.items()
    }
    loss_gradient_failures = {
        name: (
            item["relative_l2"] >= _MAX_RELATIVE_L2
            or item["cosine"] < _MIN_COSINE
            or item["max_abs"] > _MAX_ABSOLUTE_ERROR
        )
        for name, item in ignored_loss_gradient_metrics.items()
    }
    wrong_dq_tail = ignored_loss_mask[1][:, valid_tokens:]
    ignored_key_gradient_exact_equal = {
        name: bool(np.array_equal(wrong, correct))
        for name, wrong, correct in zip(
            ("dq", "dk", "dv"), ignored_key_mask[1:], expected[1:], strict=True
        )
    }
    ignored_loss_output_exact_equal = bool(
        np.array_equal(ignored_loss_mask[0], expected_output)
    )
    return {
        "ignored_key_mask": {
            "comparison": "output_query_rows_385_through_511_and_boundary_rows_384_385_511",
            "affected_rows_metrics": affected_output_metrics,
            "boundary_rows_metrics": boundary_output_metrics,
            "fails_affected_output_numerical_gate": affected_key_mask_decisive,
            "boundary_controls_fail_individual_output_numerical_gates": (
                boundary_key_mask_decisive
            ),
            "gradients_exactly_equal_to_correct_reference": (
                ignored_key_gradient_exact_equal
            ),
            "backward_sensitivity_claimed": False,
        },
        "ignored_loss_mask": {
            "comparison": "full_dq_dk_dv_and_dq_query_rows_385_through_511",
            "gradient_metrics": ignored_loss_gradient_metrics,
            "full_gradient_numerical_gate_failures": loss_gradient_failures,
            "wrong_dq_padding_bitwise_positive_zero": _bitwise_positive_zero(
                np, wrong_dq_tail
            ),
            "fails_padded_dq_zero_gate": not _bitwise_positive_zero(np, wrong_dq_tail),
            "output_exactly_equal_to_correct_reference": (
                ignored_loss_output_exact_equal
            ),
        },
        "alternative_reference_sha256": hashes,
    }


def _case_calibration(
    case: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, float], float]:
    if case == _ALL_VALID_CASE:
        return (
            _EXPECTED_INPUT_SHA256,
            _EXPECTED_REFERENCE_SHA256,
            _EXPECTED_REFERENCE_NORMS,
            _EXPECTED_MAXIMUM_ABSOLUTE_VALID_LOGIT,
        )
    if case == _VALID385_CASE:
        return (
            _EXPECTED_VALID385_INPUT_SHA256,
            _EXPECTED_VALID385_REFERENCE_SHA256,
            _EXPECTED_VALID385_REFERENCE_NORMS,
            _EXPECTED_VALID385_MAXIMUM_ABSOLUTE_VALID_LOGIT,
        )
    raise RuntimeError("VJP host construction case is outside the closed set")


def _construct_host_case(
    np: Any, ml_dtypes: Any, case: str = _ALL_VALID_CASE, *, _skip_pins: bool = False
) -> tuple[tuple[Any, ...], list[dict[str, Any]], tuple[Any, ...], dict[str, Any]]:
    case = _normalize_case(case)
    valid_tokens = _CASE_VALID_TOKENS[case]
    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    input_rng = np.random.Generator(np.random.PCG64(_INPUT_SEED))
    q = _iid_nonzero_grid(
        np, ml_dtypes, input_rng, q_shape, _QKV_MAXIMUM_INTEGER_MAGNITUDE
    )
    k = _iid_nonzero_grid(
        np, ml_dtypes, input_rng, kv_shape, _QKV_MAXIMUM_INTEGER_MAGNITUDE
    )
    v = _iid_nonzero_grid(
        np, ml_dtypes, input_rng, kv_shape, _QKV_MAXIMUM_INTEGER_MAGNITUDE
    )
    key_mask = np.zeros((_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=np.int32)
    key_mask[:, :valid_tokens] = 1
    dout_rng = np.random.Generator(np.random.PCG64(_COTANGENT_SEED))
    unmasked_dout = _iid_nonzero_grid(
        np,
        ml_dtypes,
        dout_rng,
        q_shape,
        _COTANGENT_MAXIMUM_INTEGER_MAGNITUDE,
    )
    dout = unmasked_dout.copy()
    if valid_tokens < _SEQUENCE_LENGTH:
        dout[:, valid_tokens:] = ml_dtypes.bfloat16(0)
    unmasked_dout_manifest = _array_manifest("unmasked_dout", unmasked_dout)
    inputs = (q, k, v, key_mask, dout)
    expected, oracle = _tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, key_mask, dout, scale=_ATTENTION_SCALE
    )
    input_manifests = [
        _array_manifest(name, value)
        for name, value in zip(("q", "k", "v", "key_mask", "dout"), inputs, strict=True)
    ]
    expected_manifests = [
        _array_manifest(name, value)
        for name, value in zip(("output", "dq", "dk", "dv"), expected, strict=True)
    ]
    norms = {
        name: float(np.linalg.norm(np.asarray(value, np.float64).ravel()))
        for name, value in zip(("output", "dq", "dk", "dv"), expected, strict=True)
    }
    if not all(
        math.isfinite(norm) and norm > _MIN_REFERENCE_NORM for norm in norms.values()
    ):
        raise RuntimeError("VJP host reference contains a degenerate tensor norm")
    expected_input_hashes, expected_reference_hashes, expected_norms, expected_logit = (
        _case_calibration(case)
    )
    checks = {
        "input_hashes_pinned": {
            item["name"]: item["sha256"] for item in input_manifests
        }
        == expected_input_hashes,
        "reference_hashes_pinned": {
            item["name"]: item["sha256"] for item in expected_manifests
        }
        == expected_reference_hashes,
        "reference_norms_pinned": all(
            math.isclose(norms[name], expected_norm, rel_tol=0.0, abs_tol=1e-12)
            for name, expected_norm in expected_norms.items()
        ),
        "scratch_accounting_pinned": oracle[
            "conservative_accounted_numpy_array_scratch_bytes"
        ]
        == _EXPECTED_ORACLE_SCRATCH_BYTES,
        "maximum_valid_logit_pinned": math.isclose(
            oracle["observed_maximum_absolute_valid_logit"],
            expected_logit,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    sensitivity: dict[str, Any] | None = None
    if case == _VALID385_CASE:
        sensitivity = _valid385_sensitivity_controls(
            np, q, k, v, key_mask, dout, unmasked_dout, expected
        )
        sensitivity_metrics = {
            "ignored_key_mask_affected": sensitivity["ignored_key_mask"][
                "affected_rows_metrics"
            ],
            **{
                f"ignored_key_mask_boundary_{name}": metrics
                for name, metrics in sensitivity["ignored_key_mask"][
                    "boundary_rows_metrics"
                ].items()
            },
            **{
                f"ignored_loss_mask_{name}": metrics
                for name, metrics in sensitivity["ignored_loss_mask"][
                    "gradient_metrics"
                ].items()
            },
        }
        checks.update(
            {
                "q_k_v_hashes_equal_all_valid_case": all(
                    input_manifests[index]["sha256"] == _EXPECTED_INPUT_SHA256[name]
                    for index, name in enumerate(("q", "k", "v"))
                ),
                "raw_unmasked_dout_hash_equals_all_valid_dout": (
                    unmasked_dout_manifest["sha256"] == _EXPECTED_INPUT_SHA256["dout"]
                ),
                "raw_unmasked_dout_fully_nonzero": (
                    int(np.count_nonzero(unmasked_dout)) == unmasked_dout.size
                ),
                "masked_dout_active_prefix_byte_equal_raw": (
                    dout[:, :valid_tokens].tobytes(order="C")
                    == unmasked_dout[:, :valid_tokens].tobytes(order="C")
                ),
                "masked_dout_padding_bitwise_positive_zero": (
                    _bitwise_positive_zero(np, dout[:, valid_tokens:])
                ),
                "dq_padding_reference_bitwise_positive_zero": _bitwise_positive_zero(
                    np, expected[1][:, valid_tokens:]
                ),
                "dk_padding_reference_bitwise_positive_zero": _bitwise_positive_zero(
                    np, expected[2][:, valid_tokens:]
                ),
                "dv_padding_reference_bitwise_positive_zero": _bitwise_positive_zero(
                    np, expected[3][:, valid_tokens:]
                ),
                "ignored_key_mask_control_decisive": sensitivity["ignored_key_mask"][
                    "fails_affected_output_numerical_gate"
                ]
                and all(
                    sensitivity["ignored_key_mask"][
                        "boundary_controls_fail_individual_output_numerical_gates"
                    ].values()
                ),
                "ignored_key_mask_gradient_limitation_proven": all(
                    sensitivity["ignored_key_mask"][
                        "gradients_exactly_equal_to_correct_reference"
                    ].values()
                ),
                "ignored_loss_mask_control_decisive": sensitivity["ignored_loss_mask"][
                    "fails_padded_dq_zero_gate"
                ]
                and all(
                    sensitivity["ignored_loss_mask"][
                        "full_gradient_numerical_gate_failures"
                    ].values()
                ),
                "ignored_loss_mask_output_limitation_proven": sensitivity[
                    "ignored_loss_mask"
                ]["output_exactly_equal_to_correct_reference"],
                "sensitivity_hashes_pinned": sensitivity["alternative_reference_sha256"]
                == _EXPECTED_VALID385_SENSITIVITY_SHA256,
                "sensitivity_metrics_pinned": sensitivity_metrics
                == _EXPECTED_VALID385_SENSITIVITY_METRICS,
            }
        )
    if not _skip_pins and not all(checks.values()):
        raise RuntimeError("VJP host input or oracle calibration pin changed")
    reference_manifest = {
        "outputs": expected_manifests,
        "reference_l2_norms": norms,
        "oracle": oracle,
        "calibration_pin_checks": checks,
    }
    if case == _VALID385_CASE:
        reference_manifest.update(
            {
                "case": case,
                "valid_tokens": valid_tokens,
                "padded_rows": [valid_tokens, _SEQUENCE_LENGTH],
                "sensitivity_controls": sensitivity,
                "host_loss_mask_transformation": {
                    "raw_unmasked_dout_sha256": unmasked_dout_manifest["sha256"],
                    "raw_unmasked_dout_elements": int(unmasked_dout.size),
                    "raw_unmasked_dout_nonzero_elements": int(
                        np.count_nonzero(unmasked_dout)
                    ),
                    "active_prefix_byte_equal_to_raw": (
                        dout[:, :valid_tokens].tobytes(order="C")
                        == unmasked_dout[:, :valid_tokens].tobytes(order="C")
                    ),
                    "padded_tail_bitwise_positive_zero": _bitwise_positive_zero(
                        np, dout[:, valid_tokens:]
                    ),
                    "raw_values_emitted": False,
                },
            }
        )
    return (
        inputs,
        input_manifests,
        expected,
        reference_manifest,
    )


def _tensor_metrics(
    np: Any,
    actual_host: Any,
    expected_host: Any,
    *,
    expected_shape: tuple[int, ...],
    expected_actual_nbytes: int,
) -> dict[str, Any]:
    actual_raw = np.asarray(actual_host)
    expected_raw = np.asarray(expected_host)
    actual = actual_raw.astype(np.float32)
    expected = expected_raw.astype(np.float32)
    if tuple(actual.shape) != expected_shape or tuple(expected.shape) != expected_shape:
        raise RuntimeError("candidate or VJP reference returned the wrong shape")
    actual64 = actual.ravel().astype(np.float64)
    expected64 = expected.ravel().astype(np.float64)
    difference64 = actual64 - expected64
    actual_norm = float(np.linalg.norm(actual64))
    expected_norm = float(np.linalg.norm(expected64))
    denominator = max(expected_norm, float(np.finfo(np.float64).tiny))
    cosine_denominator = max(
        actual_norm * expected_norm, float(np.finfo(np.float64).tiny)
    )
    cosine_raw = float(np.vdot(actual64, expected64) / cosine_denominator)
    return {
        "finite": bool(
            np.all(np.isfinite(actual))
            and np.all(np.isfinite(expected))
            and np.all(np.isfinite(difference64))
        ),
        "shape_dtype_nbytes_exact": (
            str(actual_raw.dtype) == "bfloat16"
            and str(expected_raw.dtype) == "float32"
            and int(actual_raw.nbytes) == expected_actual_nbytes
            and int(expected_raw.nbytes) == 2 * expected_actual_nbytes
        ),
        "actual_shape": list(actual_raw.shape),
        "actual_dtype": str(actual_raw.dtype),
        "reference_dtype": str(expected_raw.dtype),
        "actual_nbytes": int(actual_raw.nbytes),
        "reference_nbytes": int(expected_raw.nbytes),
        "reference_l2_norm": expected_norm,
        "max_abs": float(np.max(np.abs(difference64))),
        "mean_abs": float(np.mean(np.abs(difference64))),
        "relative_l2": float(np.linalg.norm(difference64) / denominator),
        "cosine_raw": cosine_raw,
        "cosine": float(np.clip(cosine_raw, -1.0, 1.0)),
        "actual_sha256": hashlib.sha256(actual_raw.tobytes(order="C")).hexdigest(),
        "reference_sha256": hashlib.sha256(expected_raw.tobytes(order="C")).hexdigest(),
    }


def _json_safe_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            key: None
            if isinstance(value, float) and not math.isfinite(value)
            else value
            for key, value in values.items()
        }
        for name, values in metrics.items()
    }


def _numerical_metrics_pass(item: dict[str, Any]) -> bool:
    return bool(
        item["finite"]
        and item["shape_dtype_nbytes_exact"]
        and item["reference_l2_norm"] > _MIN_REFERENCE_NORM
        and math.isfinite(item["relative_l2"])
        and item["relative_l2"] < _MAX_RELATIVE_L2
        and math.isfinite(item["cosine_raw"])
        and item["cosine"] >= _MIN_COSINE
        and math.isfinite(item["max_abs"])
        and item["max_abs"] <= _MAX_ABSOLUTE_ERROR
    )


def _exact_numeric_zero_manifest(np: Any, value: Any) -> dict[str, Any]:
    raw = np.asarray(value)
    fp32 = raw.astype(np.float32)
    finite = bool(np.all(np.isfinite(fp32)))
    count_nonzero = int(np.count_nonzero(raw))
    maximum_absolute_value = float(np.max(np.abs(fp32))) if raw.size else 0.0
    return {
        "finite": finite,
        "count_nonzero": count_nonzero,
        "maximum_absolute_value": maximum_absolute_value,
        "numeric_exact_zero": (
            finite and count_nonzero == 0 and maximum_absolute_value == 0.0
        ),
        "bitwise_positive_zero_diagnostic_only": _bitwise_positive_zero(np, raw),
    }


def _validate_candidate(
    np: Any,
    actual_host: Any,
    expected_host: tuple[Any, ...],
    seconds: float,
    counters: dict[str, int],
    output: TextIO,
    case: str = _ALL_VALID_CASE,
) -> dict[str, Any]:
    case = _normalize_case(case)
    if not isinstance(actual_host, tuple) or len(actual_host) != 4:
        raise RuntimeError("checked VJP executable did not return an exact four-tuple")
    shapes = (
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM),
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM),
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
    )
    nbytes = (4_194_304, 4_194_304, 1_048_576, 1_048_576)
    metrics = {
        name: _tensor_metrics(
            np,
            actual,
            expected,
            expected_shape=shape,
            expected_actual_nbytes=size,
        )
        for name, actual, expected, shape, size in zip(
            ("output", "dq", "dk", "dv"),
            actual_host,
            expected_host,
            shapes,
            nbytes,
            strict=True,
        )
    }
    per_tensor = {name: _numerical_metrics_pass(item) for name, item in metrics.items()}
    padded_validation: dict[str, Any] | None = None
    padded_passed = True
    if case == _VALID385_CASE:
        valid_tokens = _CASE_VALID_TOKENS[case]
        output_tail_actual = np.asarray(actual_host[0])[:, valid_tokens:]
        output_tail_expected = np.asarray(expected_host[0])[:, valid_tokens:]
        affected_output_metrics = _tensor_metrics(
            np,
            output_tail_actual,
            output_tail_expected,
            expected_shape=(
                _BATCH_SIZE,
                _SEQUENCE_LENGTH - valid_tokens,
                _QUERY_HEADS,
                _HEAD_DIM,
            ),
            expected_actual_nbytes=(
                _BATCH_SIZE
                * (_SEQUENCE_LENGTH - valid_tokens)
                * _QUERY_HEADS
                * _HEAD_DIM
                * 2
            ),
        )
        affected_output_passed = _numerical_metrics_pass(affected_output_metrics)
        boundary_rows = (valid_tokens - 1, valid_tokens, _SEQUENCE_LENGTH - 1)
        boundary_output_metrics = {
            str(row): _tensor_metrics(
                np,
                np.asarray(actual_host[0])[:, row : row + 1],
                np.asarray(expected_host[0])[:, row : row + 1],
                expected_shape=(_BATCH_SIZE, 1, _QUERY_HEADS, _HEAD_DIM),
                expected_actual_nbytes=_BATCH_SIZE * _QUERY_HEADS * _HEAD_DIM * 2,
            )
            for row in boundary_rows
        }
        boundary_output_passed = {
            row: _numerical_metrics_pass(item)
            for row, item in boundary_output_metrics.items()
        }
        active_gradient_metrics = {
            name: _tensor_metrics(
                np,
                np.asarray(actual)[:, :valid_tokens],
                np.asarray(expected)[:, :valid_tokens],
                expected_shape=(
                    _BATCH_SIZE,
                    valid_tokens,
                    heads,
                    _HEAD_DIM,
                ),
                expected_actual_nbytes=(
                    _BATCH_SIZE * valid_tokens * heads * _HEAD_DIM * 2
                ),
            )
            for name, actual, expected, heads in zip(
                ("dq", "dk", "dv"),
                actual_host[1:],
                expected_host[1:],
                (_QUERY_HEADS, _KV_HEADS, _KV_HEADS),
                strict=True,
            )
        }
        active_gradient_passed = {
            name: _numerical_metrics_pass(item)
            for name, item in active_gradient_metrics.items()
        }
        zero_tails = {
            name: {
                "actual": _exact_numeric_zero_manifest(
                    np, np.asarray(actual)[:, valid_tokens:]
                ),
                "reference": _exact_numeric_zero_manifest(
                    np, np.asarray(expected)[:, valid_tokens:]
                ),
            }
            for name, actual, expected in zip(
                ("dq", "dk", "dv"),
                actual_host[1:],
                expected_host[1:],
                strict=True,
            )
        }
        zero_tails_passed = all(
            item["actual"]["numeric_exact_zero"]
            and item["reference"]["numeric_exact_zero"]
            and item["reference"]["bitwise_positive_zero_diagnostic_only"]
            for item in zero_tails.values()
        )
        padded_passed = (
            affected_output_passed
            and all(boundary_output_passed.values())
            and all(active_gradient_passed.values())
            and zero_tails_passed
        )
        padded_validation = {
            "valid_tokens": valid_tokens,
            "padded_rows": [valid_tokens, _SEQUENCE_LENGTH],
            "affected_output_rows_metrics": _json_safe_metrics(
                {"output": affected_output_metrics}
            )["output"],
            "affected_output_rows_numerical_passed": affected_output_passed,
            "boundary_output_rows": list(boundary_rows),
            "boundary_output_rows_metrics": _json_safe_metrics(boundary_output_metrics),
            "boundary_output_rows_numerical_passed": boundary_output_passed,
            "active_gradient_metrics": _json_safe_metrics(active_gradient_metrics),
            "active_gradient_numerical_passed": active_gradient_passed,
            "exact_numeric_zero_tails": zero_tails,
            "candidate_tail_sign_bits_are_diagnostic_only": True,
            "all_zero_tail_gates_passed": zero_tails_passed,
            "passed": padded_passed,
        }
    safety_duration = math.isfinite(seconds) and 0 <= seconds < _MAX_CANDIDATE_SECONDS
    promotion_duration = (
        math.isfinite(seconds) and 0 <= seconds < _MAX_PROMOTION_CANDIDATE_SECONDS
    )
    passed = (
        all(per_tensor.values())
        and padded_passed
        and safety_duration
        and promotion_duration
    )
    record = {
        "record_type": "host_vjp_validation",
        "timestamp": _utc_now(),
        "status": (
            "passed"
            if passed
            else "not_promoted"
            if all(per_tensor.values()) and padded_passed and safety_duration
            else "failed"
        ),
        "metrics": _json_safe_metrics(metrics),
        "thresholds_per_tensor": {
            "finite_required": True,
            "minimum_reference_l2_norm": _MIN_REFERENCE_NORM,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
        },
        "duration_thresholds": {
            "candidate_total_seconds_strictly_below": _MAX_CANDIDATE_SECONDS,
            "promotion_total_seconds_strictly_below": (
                _MAX_PROMOTION_CANDIDATE_SECONDS
            ),
        },
        "gates": {
            "per_tensor_numerical_passed": per_tensor,
            "all_tensor_numerical_passed": all(per_tensor.values()),
            "case_specific_padded_validation_passed": padded_passed,
            "safety_duration_passed": safety_duration,
            "promotion_duration_passed": promotion_duration,
            "promotion_passed": passed,
        },
        "candidate_total_seconds": _json_safe_duration(seconds),
        "counters": dict(counters),
    }
    if padded_validation is not None:
        record["padded_validation"] = padded_validation
    _emit(record, output)
    if not passed:
        raise RuntimeError("full VJP candidate failed numerical or duration gates")
    return record


def _device_put_inputs(
    jax: Any, host_inputs: tuple[Any, ...], counters: dict[str, int]
) -> tuple[Any, ...]:
    counters["input_device_put_attempts"] += 1
    placed = jax.device_put(host_inputs)
    placed = jax.block_until_ready(placed)
    counters["input_device_put_completions"] += 1
    if not isinstance(placed, tuple) or len(placed) != 5:
        raise RuntimeError("explicit VJP device_put did not preserve the five-tuple")
    return placed


def _dispatch_candidate(
    jax: Any,
    executable: _CheckedVjpExecutable,
    arguments: tuple[Any, ...],
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, float]:
    start: float | None = None

    def on_started() -> None:
        nonlocal start
        _emit(
            {
                "record_type": "dispatch_started",
                "timestamp": _utc_now(),
                "label": "single_checked_t512_forward_and_full_vjp",
                "expected_internal_custom_calls": dict.fromkeys(_EXPECTED_MARKERS, 1),
                "counters": dict(counters),
            },
            output,
        )
        start = time.perf_counter()

    attempt_start = time.perf_counter()
    try:
        result = executable.invoke(jax, arguments, on_started)
    finally:
        seconds = time.perf_counter() - (start if start is not None else attempt_start)
        _journal_checkpoint(
            require_clean_boot, output, "after_candidate_dispatch_attempt", counters
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": "single_checked_t512_forward_and_full_vjp",
            "seconds": _json_safe_duration(seconds),
            "expected_internal_custom_call_count": 3,
            "counters": dict(counters),
        },
        output,
    )
    return result, seconds


def _device_get_candidate(
    jax: Any,
    value: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> Any:
    counters["device_get_attempts"] += 1
    try:
        host = jax.device_get(value)
        counters["device_get_completions"] += 1
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_candidate_device_get_attempt", counters
        )
    return host


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    compile_diagnostic: bool = False,
    case: str = _ALL_VALID_CASE,
    _dependencies: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> int:
    case = _normalize_case(case)
    proof = _prove_command_buffers_disabled(environment)
    architecture_binding = _assert_gfx1100_drm()
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "status": "passed",
            "proof": proof,
            "counters": dict(counters),
        },
        output,
    )
    try:
        if _dependencies is None:
            import jax
            import jax.numpy as jnp
            import jaxlib
            import ml_dtypes
            import numpy as np
            from jax.extend import backend as jax_backend

            from skyrl.tx.kernels.query_bounded_gqa import query_bounded_gqa
        else:
            jax, jnp, jaxlib, jax_backend, np, ml_dtypes, query_bounded_gqa = (
                _dependencies
            )
        backend = (
            _nonzero_probe()._length_probe()._backend_manifest(jax, jaxlib, jax_backend)
        )
        kernel_binding = _assert_kernel_binding(query_bounded_gqa)
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "backend": backend,
                "architecture_binding": architecture_binding,
                "kernel_binding": kernel_binding,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_backend_initialization_attempt", counters
        )

    if compile_diagnostic:
        return _run_compile_diagnostic(
            jax,
            jnp,
            query_bounded_gqa,
            require_clean_boot,
            counters,
            output,
        )

    compiled, compile_report = _compile_vjp_artifact(
        jax, jnp, query_bounded_gqa, require_clean_boot, counters, output
    )
    executable = _release_checked_vjp(compiled, compile_report, counters)
    del compiled
    counters["host_reference_construction_attempts"] += 1
    try:
        host_inputs, input_manifests, expected_host, reference_manifest = (
            _construct_host_case(np, ml_dtypes, case)
        )
        counters["host_reference_construction_completions"] += 1
        _emit(
            {
                "record_type": "host_t512_full_vjp_reference",
                "timestamp": _utc_now(),
                "construction": {
                    "q_k_v": "independent nonzero BF16 host PCG64 signed grids",
                    "dout": (
                        "separate-seed independent nonzero BF16 host PCG64 signed grid"
                        if case == _ALL_VALID_CASE
                        else "separate-seed independent nonzero BF16 host PCG64 signed grid with rows 385:512 set to bitwise positive zero"
                    ),
                    "key_mask": (
                        "all int32 ones"
                        if case == _ALL_VALID_CASE
                        else "int32 ones through token 384 and zeros through token 511"
                    ),
                    "scale_exact_fraction": "3/32",
                    "oracle": "independent host FP32 three-pass tiled causal forward/VJP",
                    "accelerator_rng_used": False,
                },
                "inputs": input_manifests,
                "reference": reference_manifest,
                "case": case,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_reference_construction", counters
        )

    try:
        inputs = _device_put_inputs(jax, host_inputs, counters)
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_explicit_input_device_put_attempt",
            counters,
        )
    actual, seconds = _dispatch_candidate(
        jax, executable, inputs, require_clean_boot, counters, output
    )
    actual_host = _device_get_candidate(
        jax, actual, require_clean_boot, counters, output
    )
    try:
        validation = _validate_candidate(
            np, actual_host, expected_host, seconds, counters, output, case
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_validation", counters
        )
    if counters != _completed_counters():
        raise RuntimeError("full VJP runtime counter contract was not exact")
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": (
                "passed_exact_t512_full_arbitrary_cotangent_vjp"
                if case == _ALL_VALID_CASE
                else "passed_exact_t512_valid385_loss_masked_active_token_cotangent_vjp"
            ),
            "case": case,
            "scale_exact_fraction": "3/32",
            "compile_release_gate": compile_report["release_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "host_validation": validation,
            "counters": dict(counters),
            "logical_internal_custom_calls": dict.fromkeys(_EXPECTED_MARKERS, 1),
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "second_vjp_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_dispatcher_connected": False,
        },
        output,
    )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    compile_diagnostic = getattr(args, "compile_diagnostic", False)
    case = _normalize_case(getattr(args, "case", _ALL_VALID_CASE))
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "compile_diagnostic": compile_diagnostic,
            "case": case,
            "scope": (
                "abstract_refusal"
                if args.platform == "abstract"
                else (
                    "guarded_exact_t512_full_vjp_compile_diagnostic"
                    if case == _ALL_VALID_CASE
                    else "guarded_exact_t512_valid385_loss_masked_full_vjp_compile_diagnostic"
                )
                if compile_diagnostic
                else (
                    "guarded_exact_t512_full_vjp"
                    if case == _ALL_VALID_CASE
                    else "guarded_exact_t512_valid385_loss_masked_active_token_cotangent_vjp"
                )
            ),
            "contract": (
                _abstract_contract()
                if args.platform == "abstract"
                else _compile_diagnostic_contract(case)
                if compile_diagnostic
                else _exact_contract(case)
            ),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "fresh_process_required": True,
            "prior_compile_artifact_used": False,
            "raw_ir_emitted": False,
            "runtime_execution_authorized": args.platform == "rocm"
            and not compile_diagnostic,
            "outer_profile_rocm_supervision_required": True,
            "outer_profile_rocm_supervision_operational_not_internally_proven": True,
            "counters": dict(counters),
            **_source_hashes(),
        },
        output,
    )
    if args.platform == "abstract":
        _emit(
            {
                "record_type": "refused",
                "timestamp": _utc_now(),
                "status": "no_gpu_abstract_manifest_only",
                "reason": (
                    "pass --platform rocm --allow-gpu --output explicitly under "
                    "profile_rocm.py in a fresh process"
                ),
                "jax_imported": False,
                "counters": dict(counters),
            },
            output,
        )
        return 0

    stage = "fresh_process_preflight"
    try:
        _assert_fresh_accelerator_process()
        stage = "source_binding_preflight"
        source_binding = _assert_static_source_bindings()
        _emit(
            {
                "record_type": "source_binding_proof",
                "timestamp": _utc_now(),
                "stage": "pinned_sources_bound_before_environment_or_jax",
                "proof": source_binding,
                "counters": dict(counters),
            },
            output,
        )
        stage = "safety_callable_binding_preflight"
        guarded_process, require_clean_boot = _load_safety_helpers()
        safety_binding = _safety_binding_manifest((guarded_process, require_clean_boot))
        if safety_binding["passed"] is not True:
            raise RuntimeError("retained safety callable binding proof did not pass")
        _emit(
            {
                "record_type": "safety_callable_binding_proof",
                "timestamp": _utc_now(),
                "stage": "validated_and_retained_before_environment_mutation",
                "proof": safety_binding,
                "counters": dict(counters),
            },
            output,
        )
        stage = "bounded_environment"
        environment = _configure_rocm_environment()
        _emit(
            {
                "record_type": "environment",
                "timestamp": _utc_now(),
                "stage": "bounded_rocm_environment_configured",
                "environment": _environment_manifest(environment),
                "counters": dict(counters),
            },
            output,
        )
        stage = "safety_preflight"
        with guarded_process() as raw_safety:
            safety = _public_safety_preflight(raw_safety)
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "stage": "guard_acquired",
                    "safety": safety,
                    "counters": dict(counters),
                },
                output,
            )
            stage = "compile_diagnostic" if compile_diagnostic else "runtime"
            try:
                result = _run_rocm(
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
                    compile_diagnostic=compile_diagnostic,
                    case=case,
                )
            finally:
                try:
                    postflight = _public_clean_safety(
                        require_clean_boot(), "safety_postflight"
                    )
                except Exception:
                    stage = "safety_postflight"
                    raise
                _emit(
                    {
                        "record_type": "safety_postflight",
                        "timestamp": _utc_now(),
                        "stage": "current_boot_rechecked",
                        "safety": postflight,
                        "counters": dict(counters),
                    },
                    output,
                )
        _emit(
            {
                "record_type": "completed",
                "timestamp": _utc_now(),
                "status": (
                    "compile_diagnostic_completed_no_runtime_release"
                    if compile_diagnostic
                    else "passed"
                ),
                "counters": dict(counters),
            },
            output,
        )
        return result
    except Exception as error:
        _emit(
            {
                "record_type": "error",
                "timestamp": _utc_now(),
                "stage": stage,
                "status": "failed_closed",
                "error_type": type(error).__name__,
                **_redacted_message_summary(error),
                "counters": dict(counters),
            },
            output,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    with _open_exclusive_output(args.output) as output:
        return _execute(args, output)


if __name__ == "__main__":
    sys.exit(main())
