#!/usr/bin/env python3
"""Guarded exact-T1024/C256 nonzero-logit scale-sentinel GQA probe.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm requires ``--platform rocm --allow-gpu --output`` in a fresh process under
``profile_rocm.py``.  The probe lowers and compiles exactly one BF16 final-chunk
forward with an explicit, exactly binary-representable scale of 3/32.  It
releases a one-shot executable only after both IR dialects prove one exact q768
ROCm Triton call with no outer while and compiled memory proves exact argument,
output, alias, and bounded temporary bytes.

Q, K, and V use deterministic host PCG64 draws for every individual feature
from a nonzero signed integer grid bounded by 96/128.  The independent host
FP32 stable-softmax oracle streams over query and key tiles; it never constructs
the full C-by-T logits or probability matrix.  A second host-only oracle at the
wrong scale 1/16 must differ by relative L2 greater than 0.05 before any device
input placement.  There is one lower, one compile, one input-tuple device_put,
one checked executable invocation, one device_get, and no warmup, replay,
backward, GPU reference/reduction, or model invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 1024
_QUERY_SIZE = 256
_QUERY_START = _SEQUENCE_LENGTH - _QUERY_SIZE
_QUERY_HEADS = 16
_KV_HEADS = 4
_HEAD_DIM = 256
_GROUP_SIZE = _QUERY_HEADS // _KV_HEADS
_BLOCK_Q = 64
_BLOCK_K = 64
_ORACLE_QUERY_TILE = 32
_ORACLE_KEY_TILE = 64
_SEED = 20260713
_GRID_DENOMINATOR = 128
_MAXIMUM_INTEGER_MAGNITUDE = 96
_ATTENTION_SCALE = 3 / 32
_WRONG_SCALE = 1 / 16
_MIN_WRONG_SCALE_RELATIVE_L2 = 0.05
_EXPECTED_ARGUMENT_BYTES = 6_295_552
_EXPECTED_OUTPUT_BYTES = 2_097_152
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MAX_DISPATCH_SECONDS = 0.1
_MAX_PROMOTION_DISPATCH_SECONDS = 0.075
_EXACT_MARKER = "query_bounded_gqa_forward_q768"
_EXPECTED_LENGTH_PROBE_SOURCE_SHA256 = (
    "64a38e3d381d8b75cae064a43c850c3d7d3d28e610631522cf52fba0a6483aa4"
)
_EXPECTED_COMPILE_HELPER_SOURCE_SHA256 = (
    "24eeed83e93da1133d2e1bc3d0065bc8369d13fa324d2157e37db8b9c4a4d12d"
)
_EXPECTED_SAFETY_HELPER_SOURCE_SHA256 = (
    "7ad79b9b9b54089add72dff65ea18505a794c51f0c4bafe231fbd3b745f23ba6"
)
_EXPECTED_KERNEL_SOURCE_SHA256 = (
    "51e2fd91eb270f7b25ecdd117d7f06aa48a8e4af282a5a7e5e6b4c2a25dc52c9"
)
_EXPECTED_SAFETY_CALLABLE_NAMES = (
    "guarded_qwen35_rocm_process",
    "require_clean_amdgpu_boot",
)
_RAW_IR_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
_COMPILE_GPU_WORK_CAVEAT = (
    "lowered.compile may dispatch bounded GPU autotuning/profiling work; the "
    "compiled executable remains inaccessible until exact structural and "
    "compiled-memory gates pass"
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
        "delegated_length_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_chunk_length.py",
        "delegated_chunk_compile_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_chunk_compile.py",
        "delegated_safety_helper_source_sha256": repo / "rocm" / "amdgpu_safety.py",
        "query_bounded_gqa_kernel_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "query_bounded_gqa.py",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in _source_files().items()}


def _length_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_chunk_length
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_chunk_length  # type: ignore[no-redef]

    return probe_query_bounded_gqa_chunk_length


def _assert_static_source_bindings() -> dict[str, Any]:
    """Bind every delegated executable source before environment or JAX setup."""
    length = _length_probe()
    files = _source_files()
    expected_length_path = files["delegated_length_probe_source_sha256"].resolve()
    loaded_length_path = getattr(length, "__file__", None)
    if (
        not isinstance(loaded_length_path, str)
        or Path(loaded_length_path).resolve() != expected_length_path
    ):
        raise RuntimeError(
            "length helper did not resolve to the exact repository source"
        )

    expected_hashes = {
        "delegated_length_probe_source_sha256": _EXPECTED_LENGTH_PROBE_SOURCE_SHA256,
        "delegated_chunk_compile_probe_source_sha256": (
            _EXPECTED_COMPILE_HELPER_SOURCE_SHA256
        ),
        "delegated_safety_helper_source_sha256": (
            _EXPECTED_SAFETY_HELPER_SOURCE_SHA256
        ),
        "query_bounded_gqa_kernel_source_sha256": _EXPECTED_KERNEL_SOURCE_SHA256,
    }
    actual_hashes = {
        name: _file_sha256(files[name].resolve()) for name in expected_hashes
    }
    mismatches = [
        name
        for name, expected in expected_hashes.items()
        if actual_hashes[name] != expected
    ]
    if mismatches:
        raise RuntimeError(
            "pinned source SHA256 mismatch: " + ",".join(sorted(mismatches))
        )
    delegated_binding = length._assert_static_source_bindings()
    if delegated_binding.get("passed") is not True:
        raise RuntimeError("delegated compile/kernel source binding did not pass")
    return {
        "passed": True,
        "length_helper_resolved_file_matches_expected": True,
        "delegated_compile_kernel_binding": delegated_binding,
        **actual_hashes,
    }


def _zero_counters() -> dict[str, int]:
    return _length_probe()._zero_counters()


def _completed_counters() -> dict[str, int]:
    return _length_probe()._completed_counters()


def _exact_contract() -> dict[str, Any]:
    q_shape = [_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_chunk_nonzero_scale_sentinel",
        "inputs": [
            {
                "name": "q_chunk",
                "shape": q_shape,
                "dtype": "bfloat16",
                "value": "host_pcg64_iid_nonzero_signed_integer_grid",
            },
            {
                "name": "k",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "host_pcg64_iid_nonzero_signed_integer_grid",
            },
            {
                "name": "v",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "host_pcg64_iid_nonzero_signed_integer_grid",
            },
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, _SEQUENCE_LENGTH],
                "dtype": "int32",
                "value": "all_ones",
            },
        ],
        "output": {"shape": q_shape, "dtype": "bfloat16"},
        "sequence_length": _SEQUENCE_LENGTH,
        "query_start": _QUERY_START,
        "query_stop": _SEQUENCE_LENGTH,
        "global_query_positions": [_QUERY_START, _SEQUENCE_LENGTH - 1],
        "seed": _SEED,
        "random_generator": "numpy.random.Generator(PCG64)",
        "iid_unit": "each individual Q/K/V feature",
        "zero_values_permitted": False,
        "integer_grid_denominator": _GRID_DENOMINATOR,
        "maximum_integer_magnitude": _MAXIMUM_INTEGER_MAGNITUDE,
        "maximum_absolute_input": _MAXIMUM_INTEGER_MAGNITUDE / _GRID_DENOMINATOR,
        "scale": {
            "candidate": _ATTENTION_SCALE,
            "candidate_exact_fraction": "3/32",
            "wrong_control": _WRONG_SCALE,
            "wrong_control_exact_fraction": "1/16",
        },
        "tiles": {"block_q": _BLOCK_Q, "block_k": _BLOCK_K},
        "compile_gate": {
            "independent_dialects": ["stablehlo", "optimized_hlo"],
            "exact_custom_call_count": 1,
            "exact_kernel_marker": _EXACT_MARKER,
            "exact_query_start_metadata_required_per_dialect": _QUERY_START,
            "exact_query_size_metadata_required_per_dialect": _QUERY_SIZE,
            "absent_or_lookalike_metadata_rejected": True,
            "raw_integer_value_suffix_prefix_and_duplicate_ambiguity_rejected": True,
            "no_outer_while": True,
            "exact_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
            "exact_output_bytes": _EXPECTED_OUTPUT_BYTES,
            "exact_alias_bytes": 0,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        },
        "reference": {
            "location": "host_numpy_only",
            "dtype": "float32",
            "algorithm": "query/key-tiled streaming stable-softmax GQA",
            "query_tile": _ORACLE_QUERY_TILE,
            "key_tile": _ORACLE_KEY_TILE,
            "full_logits_or_probability_matrix_constructed": False,
        },
        "wrong_scale_sensitivity_gate": {
            "wrong_scale": _WRONG_SCALE,
            "relative_l2_strictly_above": _MIN_WRONG_SCALE_RELATIVE_L2,
            "must_pass_before_device_put": True,
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
        "numerical_gate": {
            "finite_required": True,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_candidate_seconds_strictly_below": (
                _MAX_PROMOTION_DISPATCH_SECONDS
            ),
        },
    }


def _abstract_contract() -> dict[str, Any]:
    return {
        "operation": "query_bounded_gqa_nonzero_scale_sentinel_refusal",
        "sequence_length": _SEQUENCE_LENGTH,
        "query_size": _QUERY_SIZE,
        "candidate_scale_exact_fraction": "3/32",
        "wrong_scale_exact_fraction": "1/16",
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
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite existing output")
    return args


def _configure_rocm_environment() -> dict[str, str | None]:
    return _length_probe()._configure_rocm_environment()


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    return _length_probe()._environment_manifest(environment)


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    return _length_probe()._prove_command_buffers_disabled(environment)


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    helpers = _length_probe()._load_safety_helpers()
    if not isinstance(helpers, tuple) or len(helpers) != 2:
        raise RuntimeError(
            "safety helper loader did not return the exact callable pair"
        )
    expected_path = _source_files()["delegated_safety_helper_source_sha256"].resolve()
    for helper, expected_name in zip(
        helpers, _EXPECTED_SAFETY_CALLABLE_NAMES, strict=True
    ):
        if not callable(helper) or getattr(helper, "__name__", None) != expected_name:
            raise RuntimeError("safety helper callable identity or order changed")
        module = sys.modules.get(getattr(helper, "__module__", ""))
        loaded_path = getattr(module, "__file__", None)
        if loaded_path is None or Path(loaded_path).resolve() != expected_path:
            raise RuntimeError(
                "safety helper did not resolve to the pinned repository source"
            )
        if getattr(module, expected_name, None) is not helper:
            raise RuntimeError("safety helper is not the exact exported callable")
    if _file_sha256(expected_path) != _EXPECTED_SAFETY_HELPER_SOURCE_SHA256:
        raise RuntimeError("safety helper source changed after source preflight")
    return helpers


def _safety_binding_manifest(helpers: tuple[Any, Any]) -> dict[str, Any]:
    expected_path = _source_files()["delegated_safety_helper_source_sha256"].resolve()
    if len(helpers) != 2:
        raise RuntimeError(
            "retained safety binding does not contain exactly two callables"
        )
    names = [getattr(helper, "__name__", None) for helper in helpers]
    modules = [sys.modules.get(getattr(helper, "__module__", "")) for helper in helpers]
    module_paths = [getattr(module, "__file__", None) for module in modules]
    checks = {
        "exact_callable_names_and_order": names
        == list(_EXPECTED_SAFETY_CALLABLE_NAMES),
        "both_callables_retained": all(callable(helper) for helper in helpers),
        "both_modules_bound_to_pinned_path": all(
            isinstance(path, str) and Path(path).resolve() == expected_path
            for path in module_paths
        ),
        "both_are_exact_module_exports": all(
            module is not None and getattr(module, name, None) is helper
            for module, name, helper in zip(modules, names, helpers, strict=True)
        ),
        "source_hash_still_pinned": _file_sha256(expected_path)
        == _EXPECTED_SAFETY_HELPER_SOURCE_SHA256,
    }
    return {
        "checks": checks,
        "callable_names": names,
        "source_sha256": _EXPECTED_SAFETY_HELPER_SOURCE_SHA256,
        "retained_for_guard_and_journal_use": True,
        "passed": all(checks.values()),
    }


def _open_exclusive_output(path: Path) -> TextIO:
    return _length_probe()._open_exclusive_output(path)


def _assert_fresh_accelerator_process() -> None:
    _length_probe()._assert_fresh_accelerator_process()


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    return _length_probe()._public_clean_safety(safety, stage)


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    return _length_probe()._public_safety_preflight(safety)


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    return _length_probe()._journal_checkpoint(
        require_clean_boot, output, stage, counters
    )


def _shape_signature(jax: Any, jnp: Any) -> tuple[Any, Any, Any, Any]:
    return _length_probe()._shape_signature(jax, jnp, _SEQUENCE_LENGTH)


def _strict_query_metadata_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    """Require canonical query metadata in both independently parsed dialects."""
    dialects = [str(summary.get("dialect")) for summary in summaries]
    exact_dialects = len(summaries) == 2 and sorted(dialects) == [
        "optimized_hlo",
        "stablehlo",
    ]
    per_dialect: dict[str, Any] = {}
    for summary in summaries:
        dialect = str(summary.get("dialect"))
        metadata = summary.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        query_start = metadata.get("query_start")
        query_start = query_start if isinstance(query_start, dict) else {}
        query_size = metadata.get("query_size")
        query_size = query_size if isinstance(query_size, dict) else {}
        checks = {
            "metadata_preserved_not_absent": metadata.get("preserved") is True,
            "query_start_expected_is_768": query_start.get("expected") == _QUERY_START,
            "query_start_canonical_exact": query_start.get("exact_if_present") is True,
            "query_start_has_no_lookalike_spoof": query_start.get(
                "lookalike_token_occurrences"
            )
            == 0,
            "query_size_expected_is_256": query_size.get("expected") == _QUERY_SIZE,
            "query_size_canonical_exact": query_size.get("exact_if_present") is True,
            "query_size_has_no_lookalike_spoof": query_size.get(
                "lookalike_token_occurrences"
            )
            == 0,
        }
        per_dialect[dialect] = {"checks": checks, "passed": all(checks.values())}
    checks = {
        "exactly_one_summary_per_required_dialect": exact_dialects,
        "stablehlo_exact_metadata_preserved": exact_dialects
        and per_dialect.get("stablehlo", {}).get("passed") is True,
        "optimized_hlo_exact_metadata_preserved": exact_dialects
        and per_dialect.get("optimized_hlo", {}).get("passed") is True,
    }
    return {
        "checks": checks,
        "per_dialect": per_dialect,
        "passed": all(checks.values()),
    }


def _decode_local_mlir_hex_escapes(text: str) -> str:
    return re.sub(
        r"\\([0-9A-Fa-f]{2})",
        lambda match: chr(int(match.group(1), 16)),
        text,
    ).replace(r"\"", '"')


def _canonical_raw_metadata_field(
    decoded_text: str, name: str, expected: int
) -> dict[str, Any]:
    """Parse one exact integer metadata token without delegated regex semantics."""
    token_matches = list(_RAW_IR_NAME_TOKEN_PATTERN.finditer(decoded_text))
    exact_matches = [match for match in token_matches if match.group(0) == name]
    lookalikes = [
        match.group(0)
        for match in token_matches
        if match.group(0) != name and name in match.group(0)
    ]
    parsed_values: list[str] = []
    syntax_failures = 0
    for token_match in exact_matches:
        tail = decoded_text[token_match.end() :]
        separator = re.match(r'\s*"?\s*(?:=|:)\s*', tail)
        if separator is None:
            syntax_failures += 1
            continue
        remainder = tail[separator.end() :]
        if remainder.startswith('"'):
            closing_quote = remainder.find('"', 1)
            if closing_quote < 0:
                syntax_failures += 1
                continue
            parsed_values.append(remainder[1:closing_quote])
            continue
        value = re.match(r'[^\s,;)}\]"\\]+', remainder)
        if value is None:
            syntax_failures += 1
            continue
        parsed_values.append(value.group(0))

    canonical = str(expected)
    checks = {
        "exactly_one_canonical_name_token": len(exact_matches) == 1,
        "no_prefix_suffix_or_lookalike_name_tokens": not lookalikes,
        "exactly_one_value_parsed": len(parsed_values) == 1,
        "no_field_syntax_failure": syntax_failures == 0,
        "value_is_exact_canonical_integer_text": parsed_values == [canonical],
    }
    return {
        "expected": expected,
        "name_token_occurrences": len(exact_matches),
        "lookalike_name_token_occurrences": len(lookalikes),
        "parsed_value_occurrences": len(parsed_values),
        "noncanonical_value_occurrences": sum(
            value != canonical for value in parsed_values
        ),
        "syntax_failure_occurrences": syntax_failures,
        "raw_values_emitted": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _strict_raw_query_metadata_summary(text: str, dialect: str) -> dict[str, Any]:
    if dialect not in {"stablehlo", "optimized_hlo"}:
        raise ValueError("unsupported raw metadata dialect")
    decoded = _decode_local_mlir_hex_escapes(text)
    query_start = _canonical_raw_metadata_field(decoded, "query_start", _QUERY_START)
    query_size = _canonical_raw_metadata_field(decoded, "query_size", _QUERY_SIZE)
    return {
        "dialect": dialect,
        "raw_ir_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "raw_ir_emitted": False,
        "query_start": query_start,
        "query_size": query_size,
        "passed": query_start["passed"] and query_size["passed"],
    }


def _strict_raw_query_metadata_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    dialects = [str(summary.get("dialect")) for summary in summaries]
    exact_dialects = len(summaries) == 2 and sorted(dialects) == [
        "optimized_hlo",
        "stablehlo",
    ]
    per_dialect = {str(summary.get("dialect")): summary for summary in summaries}
    checks = {
        "exactly_one_raw_summary_per_required_dialect": exact_dialects,
        "stablehlo_raw_metadata_is_unambiguous_canonical_integer_text": (
            exact_dialects and per_dialect.get("stablehlo", {}).get("passed") is True
        ),
        "optimized_hlo_raw_metadata_is_unambiguous_canonical_integer_text": (
            exact_dialects
            and per_dialect.get("optimized_hlo", {}).get("passed") is True
        ),
    }
    return {
        "checks": checks,
        "per_dialect": per_dialect,
        "passed": all(checks.values()),
    }


def _compile_checked_chunk(
    jax: Any,
    jnp: Any,
    query_bounded_gqa_forward_chunk: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, dict[str, Any]]:
    """Compile once and release only the delegated hardened checked capability."""

    length = _length_probe()

    def forward_chunk(q_arg: Any, k_arg: Any, v_arg: Any, mask_arg: Any) -> Any:
        return query_bounded_gqa_forward_chunk(
            q_arg,
            k_arg,
            v_arg,
            mask_arg,
            query_start=_QUERY_START,
            scale=_ATTENTION_SCALE,
            block_q=_BLOCK_Q,
            block_k=_BLOCK_K,
            interpret=False,
        )

    _emit(
        {
            "record_type": "stage",
            "stage": "chunk_lower_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["lower_attempts"] += 1
    lower_start = time.perf_counter()
    try:
        lowered = jax.jit(forward_chunk).lower(*_shape_signature(jax, jnp))
        lower_seconds = time.perf_counter() - lower_start
        counters["lower_completions"] += 1
        stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = length._summarize_ir(stablehlo_text, "stablehlo", _SEQUENCE_LENGTH)
        stablehlo_raw_metadata = _strict_raw_query_metadata_summary(
            stablehlo_text, "stablehlo"
        )
        del stablehlo_text
        _emit(
            {
                "record_type": "lowered",
                "stage": "chunk_lower_complete_metadata_only",
                "timestamp": _utc_now(),
                "lower_seconds": lower_seconds,
                "stablehlo": stablehlo,
                "strict_raw_query_metadata": stablehlo_raw_metadata,
                "raw_ir_emitted": False,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_chunk_lower_attempt", counters
        )

    _emit(
        {
            "record_type": "stage",
            "stage": "chunk_compile_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["compile_attempts"] += 1
    compile_start = time.perf_counter()
    compiled = None
    release = False
    try:
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - compile_start
        counters["compile_completions"] += 1
        optimized_hlo_text = compiled.as_text()
        optimized_hlo = length._summarize_ir(
            optimized_hlo_text, "optimized_hlo", _SEQUENCE_LENGTH
        )
        optimized_hlo_raw_metadata = _strict_raw_query_metadata_summary(
            optimized_hlo_text, "optimized_hlo"
        )
        del optimized_hlo_text
        memory = length._compiled_memory(compiled)
        structural = length._structural_gate(stablehlo, optimized_hlo)
        strict_metadata = _strict_query_metadata_gate(stablehlo, optimized_hlo)
        strict_raw_metadata = _strict_raw_query_metadata_gate(
            stablehlo_raw_metadata, optimized_hlo_raw_metadata
        )
        memory_gate = length._compiled_memory_gate(memory, _SEQUENCE_LENGTH)
        proof = {
            "structural_gate_passed": structural["passed"],
            "strict_query_metadata_gate_passed": strict_metadata["passed"],
            "strict_raw_query_metadata_gate_passed": strict_raw_metadata["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "explicit_scale_exact_fraction": "3/32",
            "passed": structural["passed"]
            and strict_metadata["passed"]
            and strict_raw_metadata["passed"]
            and memory_gate["passed"],
        }
        report = {
            "record_type": "chunk_compiled",
            "stage": "nonzero_scale_chunk_compile_gate",
            "exact_kernel_marker": _EXACT_MARKER,
            "explicit_scale": _ATTENTION_SCALE,
            "explicit_scale_exact_fraction": "3/32",
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "compiled_memory": memory,
            "structural_gate": structural,
            "strict_query_metadata_gate": strict_metadata,
            "strict_raw_query_metadata_gate": strict_raw_metadata,
            "compiled_memory_gate": memory_gate,
            "release_gate": proof,
            "raw_ir_emitted": False,
            "counters": dict(counters),
        }
        _emit(report, output)
        if not proof["passed"]:
            raise RuntimeError(
                "nonzero scale executable failed the structural, strict metadata, "
                "strict raw metadata, or exact memory gate"
            )
        checked = length._wrap_checked(compiled, proof, counters)
        release = True
    finally:
        if compiled is not None and not release:
            del compiled
        _journal_checkpoint(
            require_clean_boot, output, "after_chunk_compile_attempt", counters
        )
    del lowered
    _emit(
        {
            "record_type": "compile_gate_passed",
            "timestamp": _utc_now(),
            "status": "checked_one_shot_capability_created",
            "release_gate": report["release_gate"],
            "counters": dict(counters),
        },
        output,
    )
    return checked, report


def _iid_nonzero_grid(
    np: Any,
    ml_dtypes: Any,
    rng: Any,
    shape: tuple[int, ...],
) -> Any:
    magnitudes = rng.integers(
        1, _MAXIMUM_INTEGER_MAGNITUDE + 1, size=shape, dtype=np.int16
    )
    signs = 2 * rng.integers(0, 2, size=shape, dtype=np.int16) - 1
    values = (magnitudes * signs).astype(np.float32) / np.float32(_GRID_DENOMINATOR)
    result = values.astype(ml_dtypes.bfloat16)
    if tuple(result.shape) != shape or int(np.count_nonzero(result)) != result.size:
        raise RuntimeError("PCG64 signed-grid construction produced zero or bad shape")
    return result


def _streaming_causal_gqa_oracle(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    query_start: int,
    scale: float,
    query_tile: int = _ORACLE_QUERY_TILE,
    key_tile: int = _ORACLE_KEY_TILE,
) -> tuple[Any, dict[str, Any]]:
    """Host FP32 stable softmax bounded by one query/key tile and one head."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("streaming oracle requires rank-four Q/K/V")
    batch, query_size, query_heads, head_dim = q.shape
    kv_batch, sequence_length, kv_heads, kv_dim = k.shape
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
                valid = key_positions[None, :] <= query_positions[:, None]
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
        raise RuntimeError("streaming host oracle produced non-finite output")
    # Conservative explicit-array accounting, not a measured process or allocator
    # peak.  It covers simultaneously live validity, gathered valid-logit,
    # logits/subtraction/probability buffers, tiled Q/K/V, accumulator update
    # intermediates, row state/reductions, and position arrays.  NumPy/Python
    # object overhead and implementation-internal BLAS workspace remain excluded.
    conservative_accounted_scratch_bytes = (
        4
        * (
            4 * query_tile * key_tile
            + 5 * query_tile * head_dim
            + 2 * key_tile * head_dim
            + 10 * query_tile
            + 2 * key_tile
        )
        + 4 * (2 * query_tile + 2 * key_tile)
        + query_tile * key_tile
    )
    return expected, {
        "implementation": "independent_host_fp32_query_key_tiled_streaming_stable_softmax_gqa",
        "query_tile": query_tile,
        "key_tile": key_tile,
        "one_query_head_at_a_time": True,
        "q_k_v_converted_to_fp32_by_active_tile_only": True,
        "full_logits_or_probability_matrix_constructed": False,
        "conservative_accounted_numpy_array_scratch_bytes": (
            conservative_accounted_scratch_bytes
        ),
        "scratch_accounting_includes": [
            "simultaneously_live_bool_valid_mask",
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


def _dense_small_reference(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    query_start: int,
    scale: float,
) -> Any:
    """Dense test oracle, intentionally refusing the production-size contract."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise RuntimeError("dense test reference requires rank-four Q/K/V")
    batch, query_size, query_heads, head_dim = q.shape
    _, sequence_length, kv_heads, kv_dim = k.shape
    if (
        batch != 1
        or kv_dim != head_dim
        or query_heads % kv_heads
        or query_size > 64
        or sequence_length > 128
    ):
        raise RuntimeError("dense reference is restricted to small CPU tests")
    q_fp32 = np.asarray(q, dtype=np.float32)[0]
    k_fp32 = np.asarray(k, dtype=np.float32)[0]
    v_fp32 = np.asarray(v, dtype=np.float32)[0]
    expected = np.empty(q.shape, dtype=np.float32)
    keys = np.arange(sequence_length, dtype=np.int32)
    queries = query_start + np.arange(query_size, dtype=np.int32)
    valid = keys[None, :] <= queries[:, None]
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


def _comparison_metrics(np: Any, actual: Any, expected: Any) -> dict[str, Any]:
    actual_fp64 = np.asarray(actual, dtype=np.float64)
    expected_fp64 = np.asarray(expected, dtype=np.float64)
    difference = actual_fp64 - expected_fp64
    actual_norm = float(np.linalg.norm(actual_fp64.ravel()))
    expected_norm = float(np.linalg.norm(expected_fp64.ravel()))
    denominator = max(expected_norm, float(np.finfo(np.float64).tiny))
    cosine_denominator = max(
        actual_norm * expected_norm, float(np.finfo(np.float64).tiny)
    )
    cosine_raw = float(
        np.vdot(actual_fp64.ravel(), expected_fp64.ravel()) / cosine_denominator
    )
    return {
        "finite": bool(
            np.all(np.isfinite(actual_fp64))
            and np.all(np.isfinite(expected_fp64))
            and np.all(np.isfinite(difference))
        ),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
        "cosine_raw": cosine_raw,
        "cosine": float(np.clip(cosine_raw, -1.0, 1.0)),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
    }


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return _length_probe()._array_manifest(name, value)


def _construct_host_inputs(
    np: Any, ml_dtypes: Any
) -> tuple[
    tuple[Any, Any, Any, Any],
    list[dict[str, Any]],
    Any,
    dict[str, Any],
    dict[str, Any],
]:
    rng = np.random.Generator(np.random.PCG64(_SEED))
    q = _iid_nonzero_grid(
        np,
        ml_dtypes,
        rng,
        (_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM),
    )
    k = _iid_nonzero_grid(
        np,
        ml_dtypes,
        rng,
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
    )
    v = _iid_nonzero_grid(
        np,
        ml_dtypes,
        rng,
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
    )
    key_mask = np.ones((_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=np.int32)
    expected, oracle = _streaming_causal_gqa_oracle(
        np,
        q,
        k,
        v,
        query_start=_QUERY_START,
        scale=_ATTENTION_SCALE,
    )
    wrong, wrong_oracle = _streaming_causal_gqa_oracle(
        np,
        q,
        k,
        v,
        query_start=_QUERY_START,
        scale=_WRONG_SCALE,
    )
    sensitivity = _comparison_metrics(np, wrong, expected)
    sensitivity_gate = (
        sensitivity["finite"]
        and math.isfinite(sensitivity["relative_l2"])
        and sensitivity["relative_l2"] > _MIN_WRONG_SCALE_RELATIVE_L2
    )
    if not sensitivity_gate:
        raise RuntimeError(
            "wrong-scale host control did not exceed its sensitivity gate"
        )

    inputs = (q, k, v, key_mask)
    manifests = [
        _array_manifest(name, value)
        for name, value in zip(("q_chunk", "k", "v", "key_mask"), inputs, strict=True)
    ]
    expected_manifest = {
        **_array_manifest("expected_scale_3_over_32", expected),
        "oracle": oracle,
    }
    wrong_scale_control = {
        "scale": _WRONG_SCALE,
        "scale_exact_fraction": "1/16",
        "oracle": wrong_oracle,
        "output": _array_manifest("wrong_scale_1_over_16", wrong),
        "metrics_vs_scale_3_over_32": sensitivity,
        "threshold": {"relative_l2_strictly_above": _MIN_WRONG_SCALE_RELATIVE_L2},
        "passed": True,
        "authorization_effect": "required_host_sensitivity_only",
    }
    return inputs, manifests, expected, expected_manifest, wrong_scale_control


def _validate_candidate(
    np: Any,
    actual_host: Any,
    expected_host: Any,
    seconds: float,
    counters: dict[str, int],
    output: TextIO,
) -> dict[str, Any]:
    metrics = _length_probe()._host_metrics(np, actual_host, expected_host)
    numerical = (
        metrics["finite"]
        and metrics["shape_dtype_nbytes_exact"]
        and math.isfinite(metrics["relative_l2"])
        and metrics["relative_l2"] < _MAX_RELATIVE_L2
        and math.isfinite(metrics["cosine_raw"])
        and metrics["cosine"] >= _MIN_COSINE
        and math.isfinite(metrics["max_abs"])
        and metrics["max_abs"] <= _MAX_ABSOLUTE_ERROR
    )
    safety_duration = math.isfinite(seconds) and 0 <= seconds < _MAX_DISPATCH_SECONDS
    promotion_duration = (
        math.isfinite(seconds) and 0 <= seconds < _MAX_PROMOTION_DISPATCH_SECONDS
    )
    passed = numerical and safety_duration and promotion_duration
    record = {
        "record_type": "host_validation",
        "timestamp": _utc_now(),
        "status": (
            "passed"
            if passed
            else "not_promoted"
            if numerical and safety_duration
            else "failed"
        ),
        "metrics": metrics,
        "thresholds": {
            "finite_required": True,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_candidate_seconds_strictly_below": (
                _MAX_PROMOTION_DISPATCH_SECONDS
            ),
        },
        "gates": {
            "numerical_passed": numerical,
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
            "nonzero scale candidate failed numerical, safety-duration, or promotion gate"
        )
    return record


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    _dependencies: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> int:
    length = _length_probe()
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

    executable, compile_report = _compile_checked_chunk(
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
            wrong_scale_control,
        ) = _construct_host_inputs(np, ml_dtypes)
        _emit(
            {
                "record_type": "host_nonzero_scale_reference",
                "timestamp": _utc_now(),
                "construction": {
                    "q_k_v": (
                        "every BF16 feature independently sampled by host PCG64 "
                        "from nonzero signed integer magnitudes 1..96 over 128"
                    ),
                    "key_mask": "all int32 ones",
                    "candidate_scale_exact_fraction": "3/32",
                    "oracle": "host FP32 query/key-tiled streaming stable-softmax GQA",
                    "full_logits_or_probability_matrix_constructed": False,
                    "accelerator_rng_used": False,
                    "seed": _SEED,
                },
                "inputs": input_manifests,
                "expected": expected_manifest,
                "wrong_scale_control": wrong_scale_control,
                "wrong_scale_sensitivity_passed_before_device_put": True,
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
            np, actual_host, expected_host, seconds, counters, output
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_validation", counters
        )
    if counters != _completed_counters():
        raise RuntimeError("nonzero scale runtime counter contract was not exact")
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_c256_t1024_nonzero_scale_sentinel",
            "sequence_length": _SEQUENCE_LENGTH,
            "query_start": _QUERY_START,
            "scale_exact_fraction": "3/32",
            "exact_kernel_marker": _EXACT_MARKER,
            "compile_release_gate": compile_report["release_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "wrong_scale_control": wrong_scale_control,
            "host_validation": {
                "metrics": validation["metrics"],
                "gates": validation["gates"],
                "candidate_seconds": seconds,
            },
            "counters": dict(counters),
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_dispatcher_connected": False,
        },
        output,
    )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "scope": (
                "abstract_refusal"
                if args.platform == "abstract"
                else "guarded_exact_c256_t1024_nonzero_scale_sentinel"
            ),
            "contract": (
                _abstract_contract()
                if args.platform == "abstract"
                else _exact_contract()
            ),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "fresh_process_required": True,
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
            stage = "runtime"
            try:
                result = _run_rocm(
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
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
