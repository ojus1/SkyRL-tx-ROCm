#!/usr/bin/env python3
"""Guarded exact-T1024/C256 right-padding GQA forward gate.

The default ``abstract`` mode emits only a refusal manifest and imports no JAX.
The ROCm path requires an explicit case, ``--allow-gpu``, and a private output
under ``profile_rocm.py``.  Exactly one of valid768, valid769, valid831,
valid832, valid833, or valid1023 is admitted per fresh process.

The probe pins and delegates the promoted nonzero-scale gate's exact q768
lower/compile, dual-dialect parsed and raw metadata, compiled-memory, checked
capability, safety, and one-shot machinery.  Inputs preserve its deterministic
nonzero BF16 PCG64 Q/K/V and explicit scale 3/32.  Only the int32 key mask
changes to a nonempty prefix of ones.  Query rows are never masked: padded
query positions retain the kernel's current key-only-mask semantics.

An independent host FP32 stable-softmax oracle streams one query head over
query/key tiles without a full C-by-T matrix.  Validation records whole-output
metrics, worst per-query-row metrics, and transition-adjacent rows.  A required
wrong all-valid-mask oracle is compared only on affected query rows and on the
first affected row, preventing valid1023 from being diluted by 255 unaffected
rows.  There is no multi-case GPU path, warmup, replay, backward, accelerator
reference/reduction, or model invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 1024
_QUERY_SIZE = 256
_QUERY_START = 768
_QUERY_HEADS = 16
_KV_HEADS = 4
_HEAD_DIM = 256
_GROUP_SIZE = _QUERY_HEADS // _KV_HEADS
_ORACLE_QUERY_TILE = 32
_ORACLE_KEY_TILE = 64
_ATTENTION_SCALE = 3 / 32
_EXPECTED_ARGUMENT_BYTES = 6_295_552
_EXPECTED_OUTPUT_BYTES = 2_097_152
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MAX_ROW_RELATIVE_L2 = 0.01
_MIN_ROW_COSINE = 0.9999
_MAX_ROW_ABSOLUTE_ERROR = 0.02
_MIN_WRONG_MASK_AFFECTED_RELATIVE_L2 = 0.02
_MAX_DISPATCH_SECONDS = 0.1
_MAX_PROMOTION_DISPATCH_SECONDS = 0.075
_CASES = {
    "valid768": 768,
    "valid769": 769,
    "valid831": 831,
    "valid832": 832,
    "valid833": 833,
    "valid1023": 1023,
}
_CASE_ORDER = tuple(_CASES)
_EXPECTED_NONZERO_PROBE_SOURCE_SHA256 = (
    "999e027d4cc35a8d59cc294020f8865036f8fb817a847ac38f96e36b597f74ac"
)
_EXPECTED_QKV_SHA256 = (
    "16aa12a02e88387223f000513febba987c23490016e5a1c9fe32019a862afc5d",
    "85ba4ec243a74b9a2019d30c94c4a0edf62e2204a6d3148fe551113605134841",
    "60132fd8c7733d2f02381f90d59e5a3dc7d740c09f25e1833540d7c956771b8a",
)
_EXPECTED_WRONG_ALL_VALID_SHA256 = (
    "623a578b3d6b4d96461fc2a9f2d7bf97ba4fc02e3c9791f449bae89d77193db5"
)
_EXPECTED_ORACLE_SCRATCH_BYTES = 334_400
_EXPECTED_CASE_CALIBRATION = {
    "valid768": {
        "mask_sha256": "41b4c32a488d09f1b6487b1d89e4b78d93e02dea44b0cf6dbf65fb1dd4286c53",
        "expected_sha256": "017970569e9232a1d289ef1bc084c187a3e266a46250741962f3e2008d963ec1",
        "affected_relative_l2": 0.3656457574589657,
        "first_affected_relative_l2": 0.03725907557681991,
    },
    "valid769": {
        "mask_sha256": "cbc965937f4fc2c6b9a151a0024f4283e18ae1e3781198800d58f23e31b059f3",
        "expected_sha256": "92c9ff029e1ded9b7e08f56d8e3b04e9ded198f6289221de47385fff667c5845",
        "affected_relative_l2": 0.3649149870684526,
        "first_affected_relative_l2": 0.03749316685604765,
    },
    "valid831": {
        "mask_sha256": "6c0ad7401a09409a707d995e0cc35ac1a7db579c8738262c847f9d6b8cd5b4a2",
        "expected_sha256": "8ae299a67efdb7e8abb3f4d52842e248341d9d8076de2a8e9033da8357b538f5",
        "affected_relative_l2": 0.3199291222843786,
        "first_affected_relative_l2": 0.0323933495896221,
    },
    "valid832": {
        "mask_sha256": "53c0cfc02674331795d5243e14a0c67386fc9b2d901e4d9076938aac2c3fd5d9",
        "expected_sha256": "1c01b2f7ed46b6b0a948b6873755989a572520bdb68a5de3ad31612db078c4b9",
        "affected_relative_l2": 0.31870534941703016,
        "first_affected_relative_l2": 0.03485904288737268,
    },
    "valid833": {
        "mask_sha256": "ce0382fba2e5be6f36fee68b6be584555b804b7168fbbfb311653e1aecad8a80",
        "expected_sha256": "f56f62eb56fb8e5f0b7989f04cbd548fac24142e282eda9d5334e32374962617",
        "affected_relative_l2": 0.3173608215510469,
        "first_affected_relative_l2": 0.03266446620763913,
    },
    "valid1023": {
        "mask_sha256": "b70afc4fbdb3d8248e081b57a3b0ba543c0be859ea9c82e46d969fe669abcfc9",
        "expected_sha256": "c5199af1c94d6afd600b1b6147699acf10854f43bbb42dc891170874d5d5af79",
        "affected_relative_l2": 0.03331218934435959,
        "first_affected_relative_l2": 0.03331218934435959,
    },
}
_COMPILE_GPU_WORK_CAVEAT = (
    "delegated lowered.compile may dispatch bounded GPU autotuning/profiling "
    "work; no executable is exposed until every promoted structural, raw "
    "metadata, and compiled-memory gate passes"
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _source_files() -> dict[str, Path]:
    repo = _repo_root()
    return {
        "probe_source_sha256": Path(__file__),
        "delegated_nonzero_scale_probe_source_sha256": repo
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


def _nonzero_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_nonzero_scale
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_nonzero_scale  # type: ignore[no-redef]

    return probe_query_bounded_gqa_nonzero_scale


def _assert_static_source_bindings() -> dict[str, Any]:
    nonzero = _nonzero_probe()
    expected_path = _source_files()[
        "delegated_nonzero_scale_probe_source_sha256"
    ].resolve()
    loaded_path = getattr(nonzero, "__file__", None)
    if not isinstance(loaded_path, str) or Path(loaded_path).resolve() != expected_path:
        raise RuntimeError(
            "nonzero-scale helper did not resolve to the exact repository source"
        )
    source_sha256 = _file_sha256(expected_path)
    if source_sha256 != _EXPECTED_NONZERO_PROBE_SOURCE_SHA256:
        raise RuntimeError("nonzero-scale helper source SHA256 differs from its pin")
    delegated = nonzero._assert_static_source_bindings()
    if delegated.get("passed") is not True:
        raise RuntimeError("delegated nonzero-scale source binding did not pass")
    return {
        "passed": True,
        "nonzero_scale_helper_resolved_file_matches_expected": True,
        "delegated_nonzero_scale_probe_source_sha256": source_sha256,
        "delegated_source_binding": delegated,
    }


def _zero_counters() -> dict[str, int]:
    return _nonzero_probe()._zero_counters()


def _completed_counters() -> dict[str, int]:
    return _nonzero_probe()._completed_counters()


def _valid_length(case: str) -> int:
    try:
        return _CASES[case]
    except KeyError as error:
        raise ValueError("padding case is outside the exact admitted enum") from error


def _affected_query_offsets(valid_length: int) -> list[int]:
    first = max(valid_length - _QUERY_START, 0)
    if first >= _QUERY_SIZE:
        raise ValueError("padding case does not affect the exact final query chunk")
    return list(range(first, _QUERY_SIZE))


def _transition_rows(valid_length: int) -> list[dict[str, Any]]:
    first_affected = valid_length - _QUERY_START
    candidates = (
        ("last_unaffected", first_affected - 1),
        ("first_affected", first_affected),
        ("second_affected", first_affected + 1),
    )
    return [
        {
            "role": role,
            "query_offset": offset,
            "global_query_position": _QUERY_START + offset,
            "affected_by_padding": offset >= first_affected,
        }
        for role, offset in candidates
        if 0 <= offset < _QUERY_SIZE
    ]


def _exact_contract(case: str) -> dict[str, Any]:
    valid_length = _valid_length(case)
    affected = _affected_query_offsets(valid_length)
    q_shape = [_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_chunk_right_padding",
        "case": case,
        "valid_length": valid_length,
        "inputs": [
            {
                "name": "q_chunk",
                "shape": q_shape,
                "dtype": "bfloat16",
                "value": "promoted_host_pcg64_iid_nonzero_signed_grid",
            },
            {
                "name": "k",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "promoted_host_pcg64_iid_nonzero_signed_grid",
            },
            {
                "name": "v",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "promoted_host_pcg64_iid_nonzero_signed_grid",
            },
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, _SEQUENCE_LENGTH],
                "dtype": "int32",
                "value": f"ones_before_{valid_length}_zeros_after",
            },
        ],
        "output": {"shape": q_shape, "dtype": "bfloat16"},
        "sequence_length": _SEQUENCE_LENGTH,
        "query_start": _QUERY_START,
        "query_stop": _SEQUENCE_LENGTH,
        "scale": _ATTENTION_SCALE,
        "scale_exact_fraction": "3/32",
        "mask_semantics": {
            "keys_only": True,
            "query_rows_masked": False,
            "padded_query_rows_remain_defined": True,
            "right_padding_prefix_required": True,
        },
        "affected_query_offsets": [affected[0], affected[-1]],
        "affected_query_row_count": len(affected),
        "transition_rows": _transition_rows(valid_length),
        "case_policy": {
            "allowed_cases": list(_CASE_ORDER),
            "one_case_per_fresh_process": True,
            "multi_case_gpu_path_exists": False,
        },
        "compile_gate": {
            "delegated_promoted_nonzero_scale_gate": True,
            "exact_marker": "query_bounded_gqa_forward_q768",
            "exact_query_start_metadata_per_dialect": 768,
            "exact_query_size_metadata_per_dialect": 256,
            "parsed_and_independent_raw_metadata_required": True,
            "exact_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
            "exact_output_bytes": _EXPECTED_OUTPUT_BYTES,
            "exact_alias_bytes": 0,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        },
        "reference": {
            "location": "host_numpy_only",
            "dtype": "float32",
            "algorithm": "query/key-tiled key-masked causal stable-softmax GQA",
            "query_tile": _ORACLE_QUERY_TILE,
            "key_tile": _ORACLE_KEY_TILE,
            "full_logits_or_probability_matrix_constructed": False,
        },
        "wrong_all_valid_control": {
            "comparison_rows": "affected_query_rows_only",
            "first_affected_row_checked_separately": True,
            "relative_l2_strictly_above": _MIN_WRONG_MASK_AFFECTED_RELATIVE_L2,
            "must_pass_before_device_put": True,
        },
        "numerical_gate": {
            "aggregate_relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "aggregate_minimum_cosine": _MIN_COSINE,
            "aggregate_maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
            "every_row_relative_l2_strictly_below": _MAX_ROW_RELATIVE_L2,
            "every_row_minimum_cosine": _MIN_ROW_COSINE,
            "every_row_maximum_absolute_error": _MAX_ROW_ABSOLUTE_ERROR,
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_candidate_seconds_strictly_below": (
                _MAX_PROMOTION_DISPATCH_SECONDS
            ),
        },
        "dispatch_plan": {
            "lower_calls": 1,
            "compile_calls": 1,
            "input_tuple_device_put_calls": 1,
            "checked_candidate_invocations": 1,
            "device_get_calls": 1,
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_invocations": 0,
        },
    }


def _abstract_contract() -> dict[str, Any]:
    return {
        "operation": "query_bounded_gqa_right_padding_refusal",
        "allowed_cases": list(_CASE_ORDER),
        "one_case_per_fresh_process": True,
        "multi_case_gpu_path_exists": False,
        "gpu_work_authorized": False,
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
        help="acknowledge one guarded compile and one checked candidate invocation",
    )
    parser.add_argument(
        "--case",
        choices=_CASE_ORDER,
        help="exactly one right-padding case; required only for guarded ROCm",
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
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a private JSONL artifact")
    if args.platform == "rocm" and args.case is None:
        parser.error("--platform rocm requires exactly one --case")
    if args.platform == "abstract" and args.case is not None:
        parser.error("--case is only valid with --platform rocm")
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
    return _nonzero_probe()._journal_checkpoint(
        require_clean_boot, output, stage, counters
    )


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return _nonzero_probe()._array_manifest(name, value)


def _validate_prefix_mask(np: Any, key_mask: Any, sequence_length: int) -> int:
    if not isinstance(key_mask, np.ndarray):
        raise TypeError("key_mask must be a NumPy array")
    if key_mask.shape != (1, sequence_length) or not (
        np.issubdtype(key_mask.dtype, np.bool_)
        or np.issubdtype(key_mask.dtype, np.integer)
    ):
        raise TypeError("key_mask must have shape [1,T] and boolean/integer dtype")
    mask = np.asarray(key_mask, dtype=np.int32)[0]
    if not bool(np.all((mask == 0) | (mask == 1))):
        raise ValueError("key_mask values must be exactly zero or one")
    valid_length = int(np.sum(mask, dtype=np.int64))
    if valid_length <= 0 or not bool(
        np.all(mask[:valid_length] == 1) and np.all(mask[valid_length:] == 0)
    ):
        raise ValueError("key_mask must be a nonempty prefix of ones then zeros")
    return valid_length


def _streaming_key_masked_causal_gqa_oracle(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    key_mask: Any,
    *,
    query_start: int,
    scale: float,
    query_tile: int = _ORACLE_QUERY_TILE,
    key_tile: int = _ORACLE_KEY_TILE,
) -> tuple[Any, dict[str, Any]]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("streaming oracle requires rank-four Q/K/V")
    batch, query_size, query_heads, head_dim = q.shape
    kv_batch, sequence_length, kv_heads, kv_dim = k.shape
    valid_length = _validate_prefix_mask(np, key_mask, sequence_length)
    if (
        batch != 1
        or kv_batch != batch
        or v.shape != k.shape
        or kv_dim != head_dim
        or query_heads % kv_heads
        or query_start < 0
        or query_start + query_size > sequence_length
        or query_tile <= 0
        or key_tile <= 0
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise RuntimeError("streaming oracle received arrays outside its GQA contract")

    group_size = query_heads // kv_heads
    expected = np.empty(q.shape, dtype=np.float32)
    observed_maximum_absolute_valid_logit = 0.0
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        for query_offset in range(0, query_size, query_tile):
            query_stop = min(query_offset + query_tile, query_size)
            tile_rows = query_stop - query_offset
            query_positions = query_start + np.arange(
                query_offset, query_stop, dtype=np.int32
            )
            q_block = np.asarray(
                q[0, query_offset:query_stop, query_head], dtype=np.float32
            )
            row_max = np.full((tile_rows,), -np.inf, dtype=np.float32)
            row_sum = np.zeros((tile_rows,), dtype=np.float32)
            accumulator = np.zeros((tile_rows, head_dim), dtype=np.float32)
            key_limit = int(query_positions[-1]) + 1
            for key_offset in range(0, key_limit, key_tile):
                key_stop = min(key_offset + key_tile, key_limit)
                key_positions = np.arange(key_offset, key_stop, dtype=np.int32)
                k_block = np.asarray(
                    k[0, key_offset:key_stop, kv_head], dtype=np.float32
                )
                v_block = np.asarray(
                    v[0, key_offset:key_stop, kv_head], dtype=np.float32
                )
                logits = (q_block @ k_block.T).astype(np.float32, copy=False)
                logits *= np.float32(scale)
                causal_valid = key_positions[None, :] <= query_positions[:, None]
                key_valid = np.asarray(
                    key_mask[0, key_offset:key_stop] != 0, dtype=np.bool_
                )
                valid = causal_valid & key_valid[None, :]
                valid_logits = logits[valid]
                if valid_logits.size:
                    observed_maximum_absolute_valid_logit = max(
                        observed_maximum_absolute_valid_logit,
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
                accumulator = correction[:, None] * accumulator + (
                    probabilities @ v_block
                ).astype(np.float32, copy=False)
                row_max = next_max
            expected[0, query_offset:query_stop, query_head] = (
                accumulator / row_sum[:, None]
            )

    if not bool(np.all(np.isfinite(expected))):
        raise RuntimeError("streaming key-masked oracle produced non-finite output")
    conservative_scratch_bytes = (
        4
        * (
            4 * query_tile * key_tile
            + 5 * query_tile * head_dim
            + 2 * key_tile * head_dim
            + 10 * query_tile
            + 2 * key_tile
        )
        + 4 * (2 * query_tile + 2 * key_tile)
        + 2 * query_tile * key_tile
        + key_tile
    )
    return expected, {
        "implementation": "independent_host_fp32_tiled_key_masked_causal_stable_softmax_gqa",
        "query_tile": query_tile,
        "key_tile": key_tile,
        "valid_length": valid_length,
        "key_only_mask_semantics": True,
        "query_rows_masked": False,
        "q_k_v_converted_to_fp32_by_active_tile_only": True,
        "full_logits_or_probability_matrix_constructed": False,
        "conservative_accounted_numpy_array_scratch_bytes": (
            conservative_scratch_bytes
        ),
        "scratch_accounting_includes": [
            "simultaneously_live_causal_key_and_combined_validity_masks",
            "gathered_valid_logits",
            "logits_subtraction_and_probability_buffers",
            "tiled_q_k_v_and_accumulator_update_intermediates",
            "row_state_reductions_and_position_arrays",
        ],
        "scratch_accounting_excludes": [
            "required_input_and_output_arrays",
            "python_and_numpy_object_overhead",
            "numpy_blas_internal_workspace",
            "allocator_fragmentation",
        ],
        "scratch_bytes_are_conservative_accounting_not_measured_peak": True,
        "observed_maximum_absolute_valid_logit": (
            observed_maximum_absolute_valid_logit
        ),
        "scale": float(scale),
        "accelerator_used": False,
    }


def _dense_key_masked_causal_gqa_reference(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    key_mask: Any,
    *,
    query_start: int,
    scale: float,
) -> Any:
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise RuntimeError("dense CPU reference requires rank-four Q/K/V")
    batch, query_size, query_heads, head_dim = q.shape
    _, sequence_length, kv_heads, kv_dim = k.shape
    _validate_prefix_mask(np, key_mask, sequence_length)
    if (
        batch != 1
        or kv_dim != head_dim
        or query_heads % kv_heads
        or query_size > _QUERY_SIZE
        or sequence_length > _SEQUENCE_LENGTH
    ):
        raise RuntimeError("dense reference is restricted to bounded CPU tests")
    q_fp32 = np.asarray(q, dtype=np.float32)[0]
    k_fp32 = np.asarray(k, dtype=np.float32)[0]
    v_fp32 = np.asarray(v, dtype=np.float32)[0]
    expected = np.empty(q.shape, dtype=np.float32)
    keys = np.arange(sequence_length, dtype=np.int32)
    queries = query_start + np.arange(query_size, dtype=np.int32)
    valid = (keys[None, :] <= queries[:, None]) & (
        np.asarray(key_mask[0] != 0)[None, :]
    )
    group_size = query_heads // kv_heads
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        logits = (q_fp32[:, query_head] @ k_fp32[:, kv_head].T).astype(
            np.float32, copy=False
        )
        logits *= np.float32(scale)
        logits = np.where(valid, logits, np.float32(-np.inf))
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits).astype(np.float32, copy=False)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True, dtype=np.float32)
        expected[0, :, query_head] = probabilities @ v_fp32[:, kv_head]
    return expected


def _row_metrics(np: Any, actual: Any, expected: Any) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.ndim != 4:
        raise RuntimeError("row metrics require equal rank-four output shapes")
    rows = [
        _nonzero_probe()._comparison_metrics(
            np, actual[:, offset : offset + 1], expected[:, offset : offset + 1]
        )
        for offset in range(actual.shape[1])
    ]

    def extreme(name: str, *, minimum: bool = False) -> dict[str, Any]:
        values = [float(row[name]) for row in rows]
        offset = (min if minimum else max)(
            range(len(values)), key=lambda index: values[index]
        )
        return {
            "value": values[offset],
            "query_offset": offset,
            "global_query_position": _QUERY_START + offset,
        }

    return {
        "all_rows_finite": all(row["finite"] for row in rows),
        "maximum_relative_l2": extreme("relative_l2"),
        "minimum_cosine": extreme("cosine", minimum=True),
        "maximum_absolute_error": extreme("max_abs"),
        "maximum_mean_absolute_error": extreme("mean_abs"),
    }


def _transition_metrics(
    np: Any, actual: Any, expected: Any, valid_length: int
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "metrics": _nonzero_probe()._comparison_metrics(
                np,
                actual[:, row["query_offset"] : row["query_offset"] + 1],
                expected[:, row["query_offset"] : row["query_offset"] + 1],
            ),
        }
        for row in _transition_rows(valid_length)
    ]


def _construct_host_inputs(
    np: Any, ml_dtypes: Any, case: str
) -> tuple[
    tuple[Any, Any, Any, Any],
    list[dict[str, Any]],
    Any,
    dict[str, Any],
    dict[str, Any],
]:
    valid_length = _valid_length(case)
    nonzero = _nonzero_probe()
    rng = np.random.Generator(np.random.PCG64(nonzero._SEED))
    q = nonzero._iid_nonzero_grid(
        np, ml_dtypes, rng, (_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM)
    )
    k = nonzero._iid_nonzero_grid(
        np,
        ml_dtypes,
        rng,
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
    )
    v = nonzero._iid_nonzero_grid(
        np,
        ml_dtypes,
        rng,
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
    )
    key_mask = np.zeros((_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=np.int32)
    key_mask[:, :valid_length] = 1
    inputs = (q, k, v, key_mask)
    manifests = [
        _array_manifest(name, value)
        for name, value in zip(("q_chunk", "k", "v", "key_mask"), inputs, strict=True)
    ]
    if tuple(item["sha256"] for item in manifests[:3]) != _EXPECTED_QKV_SHA256:
        raise RuntimeError("promoted deterministic Q/K/V hashes changed")

    expected, oracle = _streaming_key_masked_causal_gqa_oracle(
        np,
        q,
        k,
        v,
        key_mask,
        query_start=_QUERY_START,
        scale=_ATTENTION_SCALE,
    )
    all_valid_mask = np.ones_like(key_mask)
    wrong_all_valid, wrong_oracle = _streaming_key_masked_causal_gqa_oracle(
        np,
        q,
        k,
        v,
        all_valid_mask,
        query_start=_QUERY_START,
        scale=_ATTENTION_SCALE,
    )
    affected = _affected_query_offsets(valid_length)
    first_affected = affected[0]
    affected_metrics = nonzero._comparison_metrics(
        np, wrong_all_valid[:, affected], expected[:, affected]
    )
    whole_output_metrics = nonzero._comparison_metrics(np, wrong_all_valid, expected)
    first_affected_metrics = nonzero._comparison_metrics(
        np,
        wrong_all_valid[:, first_affected : first_affected + 1],
        expected[:, first_affected : first_affected + 1],
    )
    sensitivity_passed = (
        affected_metrics["finite"]
        and first_affected_metrics["finite"]
        and affected_metrics["relative_l2"] > _MIN_WRONG_MASK_AFFECTED_RELATIVE_L2
        and first_affected_metrics["relative_l2"] > _MIN_WRONG_MASK_AFFECTED_RELATIVE_L2
    )
    if not sensitivity_passed:
        raise RuntimeError("wrong all-valid mask did not pass localized sensitivity")

    expected_manifest = {
        **_array_manifest(f"expected_{case}", expected),
        "oracle": oracle,
    }
    wrong_control = {
        "control": "all_valid_key_mask",
        "output": _array_manifest("wrong_all_valid_output", wrong_all_valid),
        "oracle": wrong_oracle,
        "affected_query_offsets": [affected[0], affected[-1]],
        "affected_query_row_count": len(affected),
        "whole_output_metrics_informational_only": whole_output_metrics,
        "affected_rows_metrics": affected_metrics,
        "first_affected_row": {
            "query_offset": first_affected,
            "global_query_position": _QUERY_START + first_affected,
            "metrics": first_affected_metrics,
        },
        "transition_rows": _transition_metrics(
            np, wrong_all_valid, expected, valid_length
        ),
        "threshold": {
            "affected_and_first_affected_relative_l2_strictly_above": (
                _MIN_WRONG_MASK_AFFECTED_RELATIVE_L2
            )
        },
        "passed": True,
        "authorization_effect": "required_host_sensitivity_before_device_put",
    }
    calibration = _EXPECTED_CASE_CALIBRATION[case]
    checks = {
        "mask_hash_pinned": manifests[3]["sha256"] == calibration["mask_sha256"],
        "expected_hash_pinned": expected_manifest["sha256"]
        == calibration["expected_sha256"],
        "wrong_all_valid_hash_pinned": wrong_control["output"]["sha256"]
        == _EXPECTED_WRONG_ALL_VALID_SHA256,
        "scratch_accounting_pinned": oracle[
            "conservative_accounted_numpy_array_scratch_bytes"
        ]
        == _EXPECTED_ORACLE_SCRATCH_BYTES,
        "affected_sensitivity_calibration_pinned": math.isclose(
            affected_metrics["relative_l2"],
            calibration["affected_relative_l2"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "first_affected_sensitivity_calibration_pinned": math.isclose(
            first_affected_metrics["relative_l2"],
            calibration["first_affected_relative_l2"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("right-padding oracle hash or calibration pin changed")
    expected_manifest["calibration_pin_checks"] = checks
    wrong_control["calibration_pin_checks"] = checks
    return inputs, manifests, expected, expected_manifest, wrong_control


def _validate_candidate(
    np: Any,
    actual_host: Any,
    expected_host: Any,
    seconds: float,
    counters: dict[str, int],
    output: TextIO,
    case: str,
) -> dict[str, Any]:
    valid_length = _valid_length(case)
    aggregate = (
        _nonzero_probe()._length_probe()._host_metrics(np, actual_host, expected_host)
    )
    actual = np.asarray(actual_host)
    expected = np.asarray(expected_host)
    per_row = _row_metrics(np, actual, expected)
    transitions = _transition_metrics(np, actual, expected, valid_length)
    aggregate_passed = (
        aggregate["finite"]
        and aggregate["shape_dtype_nbytes_exact"]
        and math.isfinite(aggregate["relative_l2"])
        and aggregate["relative_l2"] < _MAX_RELATIVE_L2
        and math.isfinite(aggregate["cosine_raw"])
        and aggregate["cosine"] >= _MIN_COSINE
        and math.isfinite(aggregate["max_abs"])
        and aggregate["max_abs"] <= _MAX_ABSOLUTE_ERROR
    )
    row_passed = (
        per_row["all_rows_finite"]
        and per_row["maximum_relative_l2"]["value"] < _MAX_ROW_RELATIVE_L2
        and per_row["minimum_cosine"]["value"] >= _MIN_ROW_COSINE
        and per_row["maximum_absolute_error"]["value"] <= _MAX_ROW_ABSOLUTE_ERROR
    )
    safety_duration = math.isfinite(seconds) and 0 <= seconds < _MAX_DISPATCH_SECONDS
    promotion_duration = (
        math.isfinite(seconds) and 0 <= seconds < _MAX_PROMOTION_DISPATCH_SECONDS
    )
    passed = aggregate_passed and row_passed and safety_duration and promotion_duration
    record = {
        "record_type": "host_validation",
        "timestamp": _utc_now(),
        "case": case,
        "valid_length": valid_length,
        "status": (
            "passed"
            if passed
            else "not_promoted"
            if aggregate_passed and row_passed and safety_duration
            else "failed"
        ),
        "aggregate_metrics": aggregate,
        "worst_per_query_row_metrics": per_row,
        "transition_adjacent_row_metrics": transitions,
        "thresholds": {
            "aggregate_relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "aggregate_minimum_cosine": _MIN_COSINE,
            "aggregate_maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
            "every_row_relative_l2_strictly_below": _MAX_ROW_RELATIVE_L2,
            "every_row_minimum_cosine": _MIN_ROW_COSINE,
            "every_row_maximum_absolute_error": _MAX_ROW_ABSOLUTE_ERROR,
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_candidate_seconds_strictly_below": (
                _MAX_PROMOTION_DISPATCH_SECONDS
            ),
        },
        "gates": {
            "aggregate_numerical_passed": aggregate_passed,
            "every_query_row_numerical_passed": row_passed,
            "safety_duration_passed": safety_duration,
            "promotion_duration_passed": promotion_duration,
            "promotion_passed": passed,
        },
        "candidate_seconds": seconds,
        "counters": dict(counters),
    }
    _emit(record, output)
    if not passed:
        raise RuntimeError(
            "right-padding candidate failed aggregate, row-local, or duration gates"
        )
    return record


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    case: str,
    _dependencies: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> int:
    nonzero = _nonzero_probe()
    length = nonzero._length_probe()
    proof = _prove_command_buffers_disabled(environment)
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

            from skyrl.tx.kernels.query_bounded_gqa import (
                query_bounded_gqa_forward_chunk,
            )
        else:
            (
                jax,
                jnp,
                jaxlib,
                jax_backend,
                np,
                ml_dtypes,
                query_bounded_gqa_forward_chunk,
            ) = _dependencies
        backend = length._backend_manifest(jax, jaxlib, jax_backend)
        kernel_binding = length._assert_kernel_binding(
            query_bounded_gqa_forward_chunk, _SEQUENCE_LENGTH
        )
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "backend": backend,
                "kernel_binding": kernel_binding,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_backend_initialization_attempt", counters
        )

    executable, compile_report = nonzero._compile_checked_chunk(
        jax,
        jnp,
        query_bounded_gqa_forward_chunk,
        require_clean_boot,
        counters,
        output,
    )
    try:
        (
            host_inputs,
            input_manifests,
            expected_host,
            expected_manifest,
            wrong_control,
        ) = _construct_host_inputs(np, ml_dtypes, case)
        _emit(
            {
                "record_type": "host_right_padding_reference",
                "timestamp": _utc_now(),
                "case": case,
                "valid_length": _valid_length(case),
                "construction": {
                    "q_k_v": "exact promoted deterministic nonzero BF16 PCG64 inputs",
                    "key_mask": "int32 nonempty prefix of ones followed by zeros",
                    "query_mask": "none; every final-chunk query row remains defined",
                    "scale_exact_fraction": "3/32",
                    "oracle": "host FP32 query/key-tiled key-masked causal stable-softmax GQA",
                    "full_logits_or_probability_matrix_constructed": False,
                    "accelerator_rng_used": False,
                },
                "inputs": input_manifests,
                "expected": expected_manifest,
                "wrong_all_valid_control": wrong_control,
                "localized_wrong_mask_sensitivity_passed_before_device_put": True,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_reference_construction", counters
        )

    try:
        inputs = length._device_put_inputs(jax, host_inputs, counters)
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_explicit_input_device_put_attempt",
            counters,
        )
    actual, seconds = length._dispatch_candidate(
        jax, executable, inputs, require_clean_boot, counters, output
    )
    actual_host = length._device_get_candidate(
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
        raise RuntimeError("right-padding runtime counter contract was not exact")
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": f"passed_exact_c256_t1024_{case}_key_only_padding",
            "case": case,
            "valid_length": _valid_length(case),
            "query_start": _QUERY_START,
            "scale_exact_fraction": "3/32",
            "exact_kernel_marker": "query_bounded_gqa_forward_q768",
            "compile_release_gate": compile_report["release_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "wrong_all_valid_control": wrong_control,
            "host_validation": validation,
            "counters": dict(counters),
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_dispatcher_connected": False,
            "multi_case_gpu_path_used": False,
        },
        output,
    )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    case = getattr(args, "case", None)
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "scope": (
                "abstract_refusal"
                if args.platform == "abstract"
                else f"guarded_exact_c256_t1024_{case}_right_padding"
            ),
            "contract": (
                _abstract_contract()
                if args.platform == "abstract"
                else _exact_contract(case)
            ),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "fresh_process_required": True,
            "one_case_per_fresh_process": True,
            "multi_case_gpu_path_exists": False,
            "prior_compile_artifact_used": False,
            "raw_ir_emitted": False,
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
                    "pass --platform rocm --allow-gpu --case <exact-enum> "
                    "--output explicitly under profile_rocm.py in a fresh process"
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
                "stage": "promoted_sources_bound_before_environment_or_jax",
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
            stage = "runtime"
            try:
                result = _run_rocm(
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
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
                "status": "passed",
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
