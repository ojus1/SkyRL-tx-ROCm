#!/usr/bin/python3
"""Fail-closed, offline ISA verifier for the fixed W8A8 LoRA compile probe.

This program understands only the JAX 0.10.2 cache shape emitted by
``probe_w8a8_lora_forward.py``.  It does not import JAX, enumerate a device, or
open a device node.  The external tool entry binaries and directly loaded
libsnappy object are SHA-256 pinned.  Their dynamic loaders and transitive
dependencies are not independently hash-closed.

The cache's top-level 1,920-byte wrapper record contains an empty 1,904-byte
ELF plus the 16-byte timestamp/random module identifier that OpenXLA appends
to ROCm binaries.  The generated code objects and six ordered custom-kernel
thunks are nested in the final split-proto record.  Qualification therefore
pins the deterministic empty ELF while treating the caller-hash-bound module
identifier as evidence, and pins the caller-path-normalized complete executable
record, ordered thunk serializations, launch dimensions, embedded gfx1100
objects, forward resource contract, and the full-tile kernel's barrier/branch
epilogue.  The forward object must contain exactly four IU8 WMMA instructions
with the expected ``neg_lo`` operand modifier.  Two complete executable
variants are admitted atomically because fresh autotuning can select either
exact auxiliary LoRA GEMM tile; the W8 forward object is identical in both.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import selectors
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence


class VerificationError(RuntimeError):
    """The evidence did not satisfy the exact offline qualification contract."""


_SCHEMA = "skyrl.rocm.w8a8_lora_isa.v1"
_EXPECTED_SYMBOL = "skyrl_qwen35_w8a8_lora_forward"
_EXPECTED_HSACO_BYTES = 7_160
_EXPECTED_HSACO_SHA256 = (
    "87a2ae903547258a4b107fad17797147c417d8ca35cc600bc35d77e46323368f"
)
_EXPECTED_METADATA_BYTES = 292
_EXPECTED_METADATA_SHA256 = (
    "ff7c0690a45581061a2ebad55150a52e34ac419430cdee8ae60f5eb144b05576"
)
_EXPECTED_SPLIT_MANIFEST = bytes.fromhex(
    "0a1a786c612e6770752e47707545786563757461626c6550726f746f"
    "12081a060a020803100112081a060a02080410011202120012021200"
)
_EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "9ca30cff5e8ca18187c33b6894a6403038bd9d4a67e2d9aa3b6ec0558301d4ee"
)
_EXPECTED_EMPTY_WRAPPER_ELF_BYTES = 1_904
_EXPECTED_EMPTY_WRAPPER_ELF_SHA256 = (
    "bf465081edca1fa73a8d1d73e9cc0a354d22038c47f21bc9f4e418388c8fd563"
)
_EXPECTED_WRAPPER_RECORD_BYTES = 1_920
_EXPECTED_ROCM_MODULE_IDENTIFIER_BYTES = 16
_EXPECTED_NORMALIZED_EXECUTABLE_RECORD_BYTES = 52_909
_EXPECTED_NORMALIZED_EXECUTABLE_RECORD_SHA256 = (
    "94e1a986416c6b1b0b3d249b5ff41c2fc11dec215612a66c21d28a15968d49bc"
)
_EXPECTED_NORMALIZED_HLO_MODULE_BYTES = 20_600
_EXPECTED_NORMALIZED_HLO_MODULE_SHA256 = (
    "aca6770fd14a7d002ad465bfe8ac09c22f77b33b5d38ca0b2bd95c13734349d5"
)
_EXECUTABLE_VARIANT_BM16 = "lora_gemm_bm16_bn16"
_EXECUTABLE_VARIANT_BM32 = "lora_gemm_bm32_bn32"
_AUTOTUNE_CACHE_DIRECTORY = b"xla_gpu_per_fusion_autotune_cache_dir"
_NORMALIZED_AUTOTUNE_CACHE_PATH = (
    b"/tmp/skyrl-w8a8-neutral-0000000000/compiler-artifacts/jax-cache/"
    + _AUTOTUNE_CACHE_DIRECTORY
)
_EXPECTED_AUTOTUNE_PATH_BYTES = 101
_EXPECTED_AUTOTUNE_PATH_FIELD_OFFSET = 19_901
_EXPECTED_AUTOTUNE_PATH_RECORD_OFFSET = 19_905
_RUN_DIRECTORY_NAME = re.compile(r"skyrl-w8a8-(?:compile|runtime)-[0-9]{10}")
_CACHE_ENTRY_NAME = re.compile(r"jit_candidate-[0-9a-f]{64}-cache")
_EXPECTED_GPU_EXECUTABLE_FIELD_SIGNATURE = (
    (1, 2),
    (2, 2),
    (6, 2),
    (8, 2),
    (9, 2),
    (10, 2),
    (11, 2),
    *((13, 2),) * 6,
    (14, 2),
)
_EXPECTED_THUNKS = (
    {
        "annotation": "input_pad_reduce_fusion",
        "kernel": "input_pad_reduce_fusion",
        "id": 1,
        "bytes": 4_321,
        "sha256": "8d43b0bf40b73b8e37d5bf14bc0f417e533ea939210d2e8bdc122d245cef632b",
        "arguments": 3,
        "written": (0, 1, 1),
        "grid": (2, 1, 1),
        "threads": (256, 1, 1),
        "shared_memory_bytes": 0,
        "elf_bytes": 4_080,
        "elf_sha256": "06a6035fabadbc8de4d7d201fa51ad2b9383a37faa84e4a0b51d9587fa3d8c7f",
    },
    {
        "annotation": "loop_convert_fusion",
        "kernel": "loop_convert_fusion",
        "id": 2,
        "bytes": 4_176,
        "sha256": "a1758d210fb4e8b0bba3db81670d2e7b4f3330c5b9d0c3663c1c4a6c1fa9233c",
        "arguments": 3,
        "written": (0, 0, 1),
        "grid": (8, 1, 1),
        "threads": (128, 1, 1),
        "shared_memory_bytes": 0,
        "elf_bytes": 3_944,
        "elf_sha256": "8db071b2d0e93475f713c566d984a940155ed293e63adf53b62a53288fada685",
    },
    {
        "annotation": "gemm_fusion_dot_general.1",
        "kernel": "gemm_fusion_dot_general_1",
        "id": 3,
        "bytes": 7_665,
        "sha256": "2d60624dfa9c9eaf151d4d218e8114d7005e97cc4037102683dfb181db2b2cd1",
        "arguments": 3,
        "written": (0, 0, 1),
        "grid": (1, 1, 1),
        "threads": (128, 1, 1),
        "shared_memory_bytes": 8_192,
        "elf_bytes": 7_408,
        "elf_sha256": "c45a0fb7f236f7b16dbdfedb905dd116a02006c16c43d6dd687c30ccedf2eaf1",
    },
    {
        "annotation": "loop_select_fusion",
        "kernel": "loop_select_fusion",
        "id": 4,
        "bytes": 3_604,
        "sha256": "7ac3140403063e8050d1e6ac41d3b0936badc61fb37c5ccff20239adb19bec05",
        "arguments": 2,
        "written": (1, 1),
        "grid": (1, 1, 1),
        "threads": (16, 1, 1),
        "shared_memory_bytes": 0,
        "elf_bytes": 3_424,
        "elf_sha256": "8e7f454a584324b303ab299d22e4a3d4ee956f29bc36c64c030256fd24068a71",
    },
    {
        "annotation": "pallas_call.1",
        "kernel": _EXPECTED_SYMBOL,
        "id": 5,
        "bytes": 7_606,
        "sha256": "70cbc851866ea28d1bf542da9ea59dd2780abed6283a47f76370c3595166603e",
        "arguments": 8,
        "written": (0, 0, 0, 0, 0, 0, 0, 1),
        "grid": (1, 2, 1),
        "threads": (128, 1, 1),
        "shared_memory_bytes": 1_024,
        "elf_bytes": _EXPECTED_HSACO_BYTES,
        "elf_sha256": _EXPECTED_HSACO_SHA256,
    },
    {
        "annotation": "wrapped_slice",
        "kernel": "wrapped_slice",
        "id": 6,
        "bytes": 3_588,
        "sha256": "2b1c407360aadc991d88b84a053c223a6521422aa45bb014adc2b29fdb971b1c",
        "arguments": 2,
        "written": (0, 1),
        "grid": (1, 1, 1),
        "threads": (51, 1, 1),
        "shared_memory_bytes": 0,
        "elf_bytes": 3_416,
        "elf_sha256": "476174a6aa35385fa65e84356f63b196540840c4ac782985b6ecf744b30c4799",
    },
)
_EXPECTED_BM32_THUNKS = (
    _EXPECTED_THUNKS[0],
    _EXPECTED_THUNKS[1],
    {
        "annotation": "gemm_fusion_dot_general.1",
        "kernel": "gemm_fusion_dot_general_1",
        "id": 3,
        "bytes": 7_922,
        "sha256": "0a6c802363f4c8dc30ddfebb195cccc66d4dadc4138db5860736c879de132a2d",
        "arguments": 3,
        "written": (0, 0, 1),
        "grid": (1, 1, 1),
        "threads": (128, 1, 1),
        "shared_memory_bytes": 16_384,
        "elf_bytes": 7_664,
        "elf_sha256": "9ab0e3abac1983fcb44279f9fe1b40da01186a6d674e271b6c103cbb69c40b2a",
    },
    *_EXPECTED_THUNKS[3:],
)
_EXPECTED_EXECUTABLE_VARIANTS = {
    _EXECUTABLE_VARIANT_BM16: {
        "normalized_executable_record_bytes": (
            _EXPECTED_NORMALIZED_EXECUTABLE_RECORD_BYTES
        ),
        "normalized_executable_record_sha256": (
            _EXPECTED_NORMALIZED_EXECUTABLE_RECORD_SHA256
        ),
        "normalized_hlo_module_bytes": _EXPECTED_NORMALIZED_HLO_MODULE_BYTES,
        "normalized_hlo_module_sha256": _EXPECTED_NORMALIZED_HLO_MODULE_SHA256,
        "thunks": _EXPECTED_THUNKS,
    },
    _EXECUTABLE_VARIANT_BM32: {
        "normalized_executable_record_bytes": 53_166,
        "normalized_executable_record_sha256": (
            "989798f1183a243fe074491578827e4b04bf2d0eb25ca127f0a3b06f93050b94"
        ),
        "normalized_hlo_module_bytes": 20_600,
        "normalized_hlo_module_sha256": (
            "577bdf1c685ce7553f3af8ff8ab6b2125247f2cc7a15ee47f4c0e7916277b03f"
        ),
        "thunks": _EXPECTED_BM32_THUNKS,
    },
}
_EXPECTED_NESTED_ELFS = tuple(
    (
        thunk["kernel"],
        thunk["elf_bytes"],
        thunk["elf_sha256"],
    )
    for thunk in _EXPECTED_THUNKS
)
_RIEGELI_SIGNATURE = bytes.fromhex(
    "83af70d10d884a3f00000000000000004000000000000000"
    "91bac23c9287e1a90000000000000000e19f13c0e9b1c372"
    "73000000000000000000000000000000"
)
_CACHE_NAME = re.compile(r"jit_candidate-([0-9a-f]{64})-cache\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_MAX_COMPRESSED_CACHE = 8 << 20
_MAX_DECOMPRESSED_CACHE = 64 << 20
_MAX_SPLIT_RECORD = 16 << 20
_MAX_ELF = 16 << 20
_MAX_RECORD_COUNT = 5
_MAX_RECORD_SIZES_BYTES = _MAX_RECORD_COUNT * 10
_MAX_NESTED_ELFS = 32
_MAX_ELF_SYMBOLS = 256
_MAX_ELF_NAME_BYTES = 256
_MAX_TOOL_STDOUT = 16 << 20
_MAX_TOOL_STDERR = 64 << 10

_ZSTDCAT = Path("/usr/bin/zstdcat")
_ZSTD_RESOLVED = Path("/usr/bin/zstd")
_ZSTD_SHA256 = "7c5468b370f7c47eda07281e3437fafc568f95d10420051e3aa522709f9342c5"
_SNAPPY = Path("/usr/lib/x86_64-linux-gnu/libsnappy.so.1")
_SNAPPY_RESOLVED = Path("/usr/lib/x86_64-linux-gnu/libsnappy.so.1.1.10")
_SNAPPY_SHA256 = "955acc8eec21bea1fe890ebf75e73d1c94438264d55e042c10ad1a7a57c2e065"
_LLVM_READOBJ = Path("/opt/rocm-7.2.4/llvm/bin/llvm-readobj")
_LLVM_READOBJ_RESOLVED = Path("/opt/rocm-7.2.4/lib/llvm/bin/llvm-readobj")
_LLVM_READOBJ_SHA256 = (
    "706037a429eafdea75862c8c46624e64294bb3554357bc6f7b2fdba6e232fa02"
)
_LLVM_OBJDUMP = Path("/opt/rocm-7.2.4/llvm/bin/llvm-objdump")
_LLVM_OBJDUMP_RESOLVED = Path("/opt/rocm-7.2.4/lib/llvm/bin/llvm-objdump")
_LLVM_OBJDUMP_SHA256 = (
    "e5bf27bb6ba178b4de94ac0d5da760b628672cd00d2ffeb40a4372fa6ad25140"
)

_EXPECTED_RESOURCES = {
    "group_segment_fixed_size": 0,
    "kernarg_segment_align": 8,
    "kernarg_segment_size": 64,
    "max_flat_workgroup_size": 128,
    "max_num_workgroups_x": 1,
    "max_num_workgroups_y": 2,
    "max_num_workgroups_z": 1,
    "private_segment_fixed_size": 0,
    "sgpr_count": 34,
    "sgpr_spill_count": 0,
    "uniform_work_group_size": 1,
    "vgpr_count": 105,
    "vgpr_spill_count": 0,
    "wavefront_size": 32,
    "workgroup_processor_mode": 1,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _variant_contract_sha256(name: str, contract: dict[str, Any]) -> str:
    payload = {
        "executable_variant": name,
        "normalized_executable_record_bytes": contract[
            "normalized_executable_record_bytes"
        ],
        "normalized_executable_record_sha256": contract[
            "normalized_executable_record_sha256"
        ],
        "normalized_hlo_module_bytes": contract["normalized_hlo_module_bytes"],
        "normalized_hlo_module_sha256": contract["normalized_hlo_module_sha256"],
        "thunks": contract["thunks"],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(encoded)


def _select_executable_variant(
    normalized_executable: bytes, normalized_hlo_module: bytes
) -> tuple[str, dict[str, Any]]:
    executable_bytes = len(normalized_executable)
    executable_sha256 = _sha256(normalized_executable)
    hlo_bytes = len(normalized_hlo_module)
    hlo_sha256 = _sha256(normalized_hlo_module)
    matches = [
        (name, contract)
        for name, contract in _EXPECTED_EXECUTABLE_VARIANTS.items()
        if executable_bytes == contract["normalized_executable_record_bytes"]
        and executable_sha256 == contract["normalized_executable_record_sha256"]
        and hlo_bytes == contract["normalized_hlo_module_bytes"]
        and hlo_sha256 == contract["normalized_hlo_module_sha256"]
    ]
    if len(matches) != 1:
        raise VerificationError(
            "normalized GPU executable record is not one exact full-tile variant"
        )
    return matches[0]


def _read_varint(data: bytes, offset: int, *, label: str) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            if offset - start > 1 and byte == 0:
                raise VerificationError(f"{label} uses a non-canonical varint")
            if value > (1 << 64) - 1:
                raise VerificationError(f"{label} overflows uint64")
            return value, offset
        shift += 7
    raise VerificationError(f"{label} is truncated or oversized")


def _wire_fields(
    data: bytes, *, label: str, max_fields: int
) -> list[tuple[int, int, Any]]:
    if max_fields <= 0:
        raise ValueError("max_fields must be positive")
    fields: list[tuple[int, int, Any]] = []
    offset = 0
    while offset < len(data):
        if len(fields) >= max_fields:
            raise VerificationError(f"{label} exceeds its field-count bound")
        key, offset = _read_varint(data, offset, label=f"{label} field key")
        field_number, wire_type = key >> 3, key & 7
        if field_number == 0:
            raise VerificationError(f"{label} has protobuf field zero")
        if wire_type == 0:
            value, offset = _read_varint(
                data, offset, label=f"{label} field {field_number}"
            )
        elif wire_type == 1:
            if offset + 8 > len(data):
                raise VerificationError(f"{label} fixed64 field is truncated")
            value = data[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            size, offset = _read_varint(
                data, offset, label=f"{label} field {field_number} length"
            )
            if size > len(data) - offset:
                raise VerificationError(f"{label} length-delimited field is truncated")
            value = data[offset : offset + size]
            offset += size
        elif wire_type == 5:
            if offset + 4 > len(data):
                raise VerificationError(f"{label} fixed32 field is truncated")
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise VerificationError(f"{label} uses unsupported wire type {wire_type}")
        fields.append((field_number, wire_type, value))
    return fields


def _path_has_no_symlink(path: Path) -> None:
    if not path.is_absolute():
        raise VerificationError("input and output paths must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if current == path:
                return
            raise VerificationError(
                f"path component does not exist: {current}"
            ) from None
        if stat.S_ISLNK(info.st_mode):
            raise VerificationError(f"symlink path component is forbidden: {current}")


def _open_private_input(path: Path) -> tuple[int, os.stat_result, dict[str, Any]]:
    _path_has_no_symlink(path)
    match = _CACHE_NAME.fullmatch(path.name)
    if match is None:
        raise VerificationError("cache filename is not an exact jit_candidate key")
    parent_info = os.stat(path.parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or parent_info.st_mode & 0o077
    ):
        raise VerificationError(
            "cache parent must be a private directory owned by this uid"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise VerificationError("cache input is not a regular file")
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise VerificationError("cache input is not private to this uid")
        if info.st_nlink != 1:
            raise VerificationError("cache input must have exactly one hard link")
        if info.st_size <= 0 or info.st_size > _MAX_COMPRESSED_CACHE:
            raise VerificationError("compressed cache size is outside the fixed bound")
        manifest = {
            "path": str(path),
            "cache_key": match.group(1),
            "bytes": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
            "uid": info.st_uid,
            "device": info.st_dev,
            "inode": info.st_ino,
            "link_count": info.st_nlink,
        }
        return descriptor, info, manifest
    except BaseException:
        os.close(descriptor)
        raise


def _read_fd(descriptor: int, maximum: int, *, label: str) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1 << 20, maximum + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise VerificationError(f"{label} exceeds its byte bound")
    return b"".join(chunks)


def _attest_fixed_file(
    alias: Path, resolved: Path, expected_sha256: str
) -> dict[str, Any]:
    try:
        actual_resolved = alias.resolve(strict=True)
    except OSError as error:
        raise VerificationError(
            f"fixed dependency is unavailable: {alias}: {error}"
        ) from error
    if actual_resolved != resolved:
        raise VerificationError(
            f"fixed dependency {alias} resolved to {actual_resolved}, expected {resolved}"
        )
    descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise VerificationError(
                f"fixed dependency is not a protected root file: {resolved}"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise VerificationError(f"fixed dependency hash changed: {resolved}")
        return {
            "path": str(alias),
            "resolved_path": str(resolved),
            "bytes": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
            "uid": info.st_uid,
            "sha256": observed,
        }
    finally:
        os.close(descriptor)


def _attest_toolchain() -> dict[str, Any]:
    return {
        "zstdcat": _attest_fixed_file(_ZSTDCAT, _ZSTD_RESOLVED, _ZSTD_SHA256),
        "libsnappy": _attest_fixed_file(_SNAPPY, _SNAPPY_RESOLVED, _SNAPPY_SHA256),
        "llvm_readobj": _attest_fixed_file(
            _LLVM_READOBJ, _LLVM_READOBJ_RESOLVED, _LLVM_READOBJ_SHA256
        ),
        "llvm_objdump": _attest_fixed_file(
            _LLVM_OBJDUMP, _LLVM_OBJDUMP_RESOLVED, _LLVM_OBJDUMP_SHA256
        ),
    }


def _run_bounded(
    command: Sequence[str],
    *,
    maximum_stdout: int,
    timeout_seconds: float,
    pass_fds: Sequence[int] = (),
) -> tuple[bytes, bytes]:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=tuple(pass_fds),
        env={"LC_ALL": "C", "LANG": "C", "PATH": "/usr/bin:/bin"},
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    streams = {
        process.stdout.fileno(): ("stdout", maximum_stdout),
        process.stderr.fileno(): ("stderr", _MAX_TOOL_STDERR),
    }
    for descriptor in streams:
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ)
    output = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VerificationError(f"fixed command timed out: {command[0]}")
            for key, _ in selector.select(min(remaining, 0.25)):
                descriptor = key.fd
                try:
                    chunk = os.read(descriptor, 1 << 16)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                name, maximum = streams[descriptor]
                output[name].extend(chunk)
                if len(output[name]) > maximum:
                    raise VerificationError(f"fixed command {name} exceeded its bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VerificationError(f"fixed command timed out: {command[0]}")
        return_code = process.wait(timeout=remaining)
        if return_code != 0:
            message = bytes(output["stderr"]).decode("utf-8", "replace").strip()
            raise VerificationError(
                f"fixed command failed with exit {return_code}: {command[0]}: {message}"
            )
        if output["stderr"]:
            raise VerificationError(
                f"fixed command produced unexpected stderr: {command[0]}"
            )
        return bytes(output["stdout"]), bytes(output["stderr"])
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()


class _Snappy:
    def __init__(self) -> None:
        self._library = ctypes.CDLL(str(_SNAPPY_RESOLVED))
        self._library.snappy_uncompressed_length.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.snappy_uncompressed_length.restype = ctypes.c_int
        self._library.snappy_uncompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.snappy_uncompress.restype = ctypes.c_int

    def decompress(self, encoded: bytes, expected_size: int) -> bytes:
        if not encoded:
            raise VerificationError("Snappy payload is empty")
        if expected_size < 0 or expected_size > _MAX_DECOMPRESSED_CACHE:
            raise VerificationError("Snappy decoded size is outside the fixed bound")
        source = ctypes.create_string_buffer(encoded, len(encoded))
        observed_size = ctypes.c_size_t()
        status = self._library.snappy_uncompressed_length(
            source, len(encoded), ctypes.byref(observed_size)
        )
        if status != 0 or observed_size.value != expected_size:
            raise VerificationError("Snappy size header does not match Riegeli")
        destination = ctypes.create_string_buffer(expected_size)
        destination_size = ctypes.c_size_t(expected_size)
        status = self._library.snappy_uncompress(
            source,
            len(encoded),
            destination,
            ctypes.byref(destination_size),
        )
        if status != 0 or destination_size.value != expected_size:
            raise VerificationError("libsnappy rejected the Riegeli block")
        return bytes(destination.raw[:expected_size])


def _decompress_riegeli_block(
    block: bytes,
    decompressor: Callable[[bytes, int], bytes],
    *,
    label: str,
    maximum: int,
) -> bytes:
    expected_size, offset = _read_varint(block, 0, label=f"{label} decoded size")
    if expected_size > maximum:
        raise VerificationError(f"{label} decoded size exceeds its fixed bound")
    if offset == len(block):
        raise VerificationError(f"{label} has no Snappy stream")
    return decompressor(block[offset:], expected_size)


def _extract_split_records(
    encoded: bytes, decompressor: Callable[[bytes, int], bytes]
) -> tuple[list[bytes], dict[str, Any]]:
    if len(encoded) < 104 or encoded[:64] != _RIEGELI_SIGNATURE:
        raise VerificationError("field 1 is not the exact Riegeli split-proto form")
    if encoded.count(_RIEGELI_SIGNATURE) != 1:
        raise VerificationError("Riegeli signature is not unique")
    header = encoded[64:104]
    data_size = int.from_bytes(header[8:16], "little")
    chunk_type = header[24]
    record_count = int.from_bytes(header[25:32], "little")
    decoded_size = int.from_bytes(header[32:40], "little")
    if data_size != len(encoded) - 104:
        raise VerificationError("Riegeli chunk size does not consume field 1 exactly")
    if chunk_type != ord("r") or record_count != 5:
        raise VerificationError("expected one simple Riegeli chunk with five records")
    if decoded_size <= 0 or decoded_size > _MAX_DECOMPRESSED_CACHE:
        raise VerificationError("Riegeli decoded size is outside the fixed bound")
    data = encoded[104:]
    if not data or data[0] != ord("s"):
        raise VerificationError("Riegeli chunk does not use exact Snappy compression")
    sizes_size, offset = _read_varint(data, 1, label="compressed sizes length")
    if sizes_size <= 0 or sizes_size > len(data) - offset:
        raise VerificationError("Riegeli compressed sizes block is invalid")
    sizes_encoded = data[offset : offset + sizes_size]
    values_encoded = data[offset + sizes_size :]
    sizes_raw = _decompress_riegeli_block(
        sizes_encoded,
        decompressor,
        label="record sizes",
        maximum=_MAX_RECORD_SIZES_BYTES,
    )
    values_raw = _decompress_riegeli_block(
        values_encoded,
        decompressor,
        label="record values",
        maximum=decoded_size,
    )
    sizes: list[int] = []
    offset = 0
    while offset < len(sizes_raw):
        size, offset = _read_varint(sizes_raw, offset, label="record size")
        sizes.append(size)
        if len(sizes) > _MAX_RECORD_COUNT:
            raise VerificationError("Riegeli record count exceeds its fixed bound")
    if len(sizes) != record_count or sum(sizes) != decoded_size:
        raise VerificationError("Riegeli record sizes do not match the chunk header")
    if len(values_raw) != decoded_size:
        raise VerificationError("Riegeli values do not match decoded_data_size")
    if sizes[:4] != [
        len(_EXPECTED_SPLIT_MANIFEST),
        0,
        _EXPECTED_WRAPPER_RECORD_BYTES,
        0,
    ]:
        raise VerificationError("split-proto fixed record sizes changed")
    if sizes[4] <= 0 or sizes[4] > _MAX_SPLIT_RECORD:
        raise VerificationError("split-proto executable record is outside its bound")
    records: list[bytes] = []
    offset = 0
    for size in sizes:
        records.append(values_raw[offset : offset + size])
        offset += size
    if records[0] != _EXPECTED_SPLIT_MANIFEST:
        raise VerificationError("split-proto manifest changed")
    return records, {
        "chunk_type": "simple",
        "compression": "snappy",
        "record_count": record_count,
        "record_sizes": sizes,
        "decoded_data_bytes": decoded_size,
        "manifest_sha256": _sha256(records[0]),
        # Riegeli uses HighwayHash internally.  This stdlib-only verifier does
        # not reimplement it; integrity is instead bound by the mandatory
        # caller-provided SHA-256 of the complete cache entry.
        "riegeli_internal_hashes_validated": False,
    }


def _extract_cache_records(
    decompressed: bytes, decompressor: Callable[[bytes, int], bytes]
) -> tuple[list[bytes], dict[str, Any]]:
    if len(decompressed) < 8 or len(decompressed) > _MAX_DECOMPRESSED_CACHE:
        raise VerificationError("decompressed cache is outside the fixed bound")
    compile_seconds = int.from_bytes(decompressed[:4], "big")
    if compile_seconds > 600:
        raise VerificationError("cached compile time is outside the microcase bound")
    body = decompressed[4:]
    metadata_size, offset = _read_varint(body, 0, label="IFRT metadata size")
    if metadata_size != _EXPECTED_METADATA_BYTES or metadata_size > len(body) - offset:
        raise VerificationError("IFRT metadata has the wrong fixed size")
    metadata = body[offset : offset + metadata_size]
    if _sha256(metadata) != _EXPECTED_METADATA_SHA256:
        raise VerificationError(
            "IFRT metadata is not the fixed jit_candidate signature"
        )
    fields = _wire_fields(
        body[offset + metadata_size :], label="PJRT executable", max_fields=2
    )
    if [(field, wire) for field, wire, _ in fields] != [(1, 2), (2, 2)]:
        raise VerificationError("PJRT executable fields are not exactly [1, 2]")
    split_proto = fields[0][2]
    compile_options = fields[1][2]
    assert isinstance(split_proto, bytes) and isinstance(compile_options, bytes)
    if not compile_options or len(compile_options) > 16 << 10:
        raise VerificationError("PJRT compile options are empty or oversized")
    records, riegeli = _extract_split_records(split_proto, decompressor)
    return records, {
        "compile_time_seconds": compile_seconds,
        "decompressed_bytes": len(decompressed),
        "ifrt_metadata_bytes": len(metadata),
        "ifrt_metadata_sha256": _sha256(metadata),
        "compile_options_bytes": len(compile_options),
        "compile_options_sha256": _sha256(compile_options),
        "riegeli": riegeli,
    }


def _field_signature(
    fields: Sequence[tuple[int, int, Any]],
) -> tuple[tuple[int, int], ...]:
    return tuple((field, wire) for field, wire, _ in fields)


def _exact_ascii(value: Any, *, label: str) -> str:
    if not isinstance(value, bytes) or not value or len(value) > _MAX_ELF_NAME_BYTES:
        raise VerificationError(f"{label} is empty, oversized, or not bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} is not ASCII") from error
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in decoded):
        raise VerificationError(f"{label} contains a non-printable character")
    return decoded


def _decode_dim3d(
    value: Any, *, wrapper_field: int, label: str
) -> tuple[int, int, int]:
    if not isinstance(value, bytes):
        raise VerificationError(f"{label} wrapper is not bytes")
    wrapper = _wire_fields(value, label=f"{label} wrapper", max_fields=1)
    if _field_signature(wrapper) != ((wrapper_field, 2),):
        raise VerificationError(f"{label} wrapper fields changed")
    encoded = wrapper[0][2]
    assert isinstance(encoded, bytes)
    dimensions = _wire_fields(encoded, label=label, max_fields=3)
    if _field_signature(dimensions) != ((1, 0), (2, 0), (3, 0)):
        raise VerificationError(f"{label} fields changed")
    result = tuple(item[2] for item in dimensions)
    if len(result) != 3 or not all(
        isinstance(item, int) and 1 <= item <= 1 << 20 for item in result
    ):
        raise VerificationError(f"{label} values are outside the fixed bound")
    return result  # type: ignore[return-value]


def _inspect_thunks(records: list[bytes], *, cache_path: Path) -> dict[str, Any]:
    """Decode and pin the complete ordered custom-kernel launch sequence."""
    if len(records) != 5:
        raise VerificationError("split-proto record count changed before thunk audit")
    executable = records[4]
    run_directory = cache_path.parent.parent.parent
    if (
        not cache_path.is_absolute()
        or cache_path.parent.name != "jax-cache"
        or cache_path.parent.parent.name != "compiler-artifacts"
        or run_directory.parent != Path("/tmp")
        or _RUN_DIRECTORY_NAME.fullmatch(run_directory.name) is None
        or _CACHE_ENTRY_NAME.fullmatch(cache_path.name) is None
    ):
        raise VerificationError(
            "cache is not under the exact fresh /tmp run-scoped artifact layout"
        )
    autotune_cache_path = os.fsencode(
        cache_path.parent / os.fsdecode(_AUTOTUNE_CACHE_DIRECTORY)
    )
    fields = _wire_fields(executable, label="GpuExecutableProto", max_fields=32)
    if _field_signature(fields) != _EXPECTED_GPU_EXECUTABLE_FIELD_SIGNATURE:
        raise VerificationError("GpuExecutableProto field sequence changed")
    hlo_module = fields[0][2]
    assert isinstance(hlo_module, bytes)
    if (
        len(_NORMALIZED_AUTOTUNE_CACHE_PATH) != _EXPECTED_AUTOTUNE_PATH_BYTES
        or len(autotune_cache_path) != _EXPECTED_AUTOTUNE_PATH_BYTES
        or _NORMALIZED_AUTOTUNE_CACHE_PATH in executable
        or executable.count(_AUTOTUNE_CACHE_DIRECTORY) != 1
        or executable.count(autotune_cache_path) != 1
        or hlo_module.count(autotune_cache_path) != 1
        or hlo_module.find(autotune_cache_path) != _EXPECTED_AUTOTUNE_PATH_FIELD_OFFSET
        or executable.find(autotune_cache_path) != _EXPECTED_AUTOTUNE_PATH_RECORD_OFFSET
    ):
        raise VerificationError(
            "executable autotune-cache path is not uniquely caller-bound"
        )
    normalized_executable = executable.replace(
        autotune_cache_path, _NORMALIZED_AUTOTUNE_CACHE_PATH
    )
    normalized_fields = _wire_fields(
        normalized_executable,
        label="normalized GpuExecutableProto",
        max_fields=32,
    )
    if _field_signature(normalized_fields) != _EXPECTED_GPU_EXECUTABLE_FIELD_SIGNATURE:
        raise VerificationError("normalized GpuExecutableProto field sequence changed")
    normalized_hlo_module = normalized_fields[0][2]
    if normalized_executable.count(
        _NORMALIZED_AUTOTUNE_CACHE_PATH
    ) != 1 or not isinstance(normalized_hlo_module, bytes):
        raise VerificationError(
            "normalized GPU executable record is not one exact full-tile variant"
        )
    executable_variant, variant_contract = _select_executable_variant(
        normalized_executable, normalized_hlo_module
    )
    expected_thunks = variant_contract["thunks"]
    if not isinstance(expected_thunks, tuple):
        raise VerificationError("internal executable variant thunk contract changed")
    encoded_thunks = [
        value for field, wire, value in fields if (field, wire) == (13, 2)
    ]
    if len(encoded_thunks) != len(expected_thunks) or not all(
        isinstance(item, bytes) for item in encoded_thunks
    ):
        raise VerificationError("ordered thunk count or wire type changed")

    inventory: list[dict[str, Any]] = []
    for index, (encoded, expected) in enumerate(
        zip(encoded_thunks, expected_thunks, strict=True)
    ):
        assert isinstance(encoded, bytes)
        label = f"ThunkProto[{index}]"
        if len(encoded) != expected["bytes"] or _sha256(encoded) != expected["sha256"]:
            raise VerificationError(f"{label} serialization changed")
        thunk_fields = _wire_fields(encoded, label=label, max_fields=2)
        if _field_signature(thunk_fields) != ((1, 2), (36, 2)):
            raise VerificationError(f"{label} is not the exact custom-kernel thunk")

        info = thunk_fields[0][2]
        custom = thunk_fields[1][2]
        assert isinstance(info, bytes) and isinstance(custom, bytes)
        info_fields = _wire_fields(info, label=f"{label}.ThunkInfoProto", max_fields=2)
        if _field_signature(info_fields) != ((1, 2), (3, 0)):
            raise VerificationError(f"{label} info fields changed")
        annotation = _exact_ascii(
            info_fields[0][2], label=f"{label} profile annotation"
        )
        thunk_id = info_fields[1][2]
        if annotation != expected["annotation"] or thunk_id != expected["id"]:
            raise VerificationError(f"{label} annotation or ID changed")

        custom_fields = _wire_fields(
            custom,
            label=f"{label}.CustomKernelThunkProto",
            max_fields=int(expected["arguments"]) + 3,
        )
        expected_custom_signature = (
            *((1, 2),) * int(expected["arguments"]),
            (2, 2),
            (3, 2),
            (6, 2),
        )
        if _field_signature(custom_fields) != expected_custom_signature:
            raise VerificationError(f"{label} argument or custom-kernel fields changed")
        arguments = [value for field, _, value in custom_fields if field == 1]
        if not all(isinstance(value, bytes) and value for value in arguments):
            raise VerificationError(f"{label} contains an empty shaped-slice argument")
        written = next(value for field, _, value in custom_fields if field == 2)
        kernel = next(value for field, _, value in custom_fields if field == 3)
        tma_metadata = next(value for field, _, value in custom_fields if field == 6)
        if (
            not isinstance(written, bytes)
            or tuple(written) != expected["written"]
            or not isinstance(kernel, bytes)
            or tma_metadata != b""
        ):
            raise VerificationError(
                f"{label} write flags, kernel, or TMA metadata changed"
            )

        kernel_fields = _wire_fields(
            kernel, label=f"{label}.CustomKernelProto", max_fields=5
        )
        expected_kernel_signature = ((1, 2), (2, 2), (3, 2), (4, 2))
        if expected["shared_memory_bytes"]:
            expected_kernel_signature += ((6, 0),)
        if _field_signature(kernel_fields) != expected_kernel_signature:
            raise VerificationError(f"{label} custom-kernel fields changed")
        by_field = {field: value for field, _, value in kernel_fields}
        kernel_name = _exact_ascii(by_field[1], label=f"{label} kernel name")
        shared_memory_bytes = int(by_field.get(6, 0))
        if (
            kernel_name != expected["kernel"]
            or shared_memory_bytes != expected["shared_memory_bytes"]
        ):
            raise VerificationError(f"{label} kernel name or LDS allocation changed")
        grid = _decode_dim3d(
            by_field[3], wrapper_field=2, label=f"{label} block dimensions"
        )
        threads = _decode_dim3d(
            by_field[4], wrapper_field=1, label=f"{label} thread dimensions"
        )
        if grid != expected["grid"] or threads != expected["threads"]:
            raise VerificationError(f"{label} launch dimensions changed")

        loader = by_field[2]
        assert isinstance(loader, bytes)
        loader_fields = _wire_fields(
            loader, label=f"{label}.KernelLoaderSpecProto", max_fields=4
        )
        if _field_signature(loader_fields) != ((2, 2), (3, 0), (4, 2), (5, 2)):
            raise VerificationError(f"{label} loader fields changed")
        loader_by_field = {field: value for field, _, value in loader_fields}
        if (
            loader_by_field[3] != expected["arguments"]
            or _exact_ascii(loader_by_field[4], label=f"{label} loader kernel name")
            != kernel_name
            or not isinstance(loader_by_field[5], bytes)
            or not loader_by_field[5]
        ):
            raise VerificationError(f"{label} loader arity/name/packing changed")
        cubin_wrapper = loader_by_field[2]
        assert isinstance(cubin_wrapper, bytes)
        cubin_fields = _wire_fields(
            cubin_wrapper, label=f"{label}.CudaCubinProto", max_fields=1
        )
        if _field_signature(cubin_fields) != ((1, 2),):
            raise VerificationError(f"{label} embedded-code fields changed")
        elf = cubin_fields[0][2]
        if (
            not isinstance(elf, bytes)
            or len(elf) != expected["elf_bytes"]
            or _sha256(elf) != expected["elf_sha256"]
        ):
            raise VerificationError(f"{label} embedded gfx1100 object changed")
        parsed_elf, elf_manifest = _parse_elf(elf, 0, source=f"{label}.embedded_elf")
        if (
            parsed_elf != elf
            or elf_manifest["global_functions"] != [kernel_name]
            or elf_manifest["global_objects"] != [f"{kernel_name}.kd"]
        ):
            raise VerificationError(f"{label} embedded object identity changed")

        inventory.append(
            {
                "order": index,
                "thunk_id": thunk_id,
                "profile_annotation": annotation,
                "kernel": kernel_name,
                "bytes": len(encoded),
                "sha256": _sha256(encoded),
                "argument_count": len(arguments),
                "written": list(written),
                "arity": loader_by_field[3],
                "grid": list(grid),
                "threads": list(threads),
                "shared_memory_bytes": shared_memory_bytes,
                "elf_bytes": len(elf),
                "elf_sha256": _sha256(elf),
                "target": elf_manifest["target"],
            }
        )
    return {
        "executable_variant": executable_variant,
        "executable_variant_contract_sha256": _variant_contract_sha256(
            executable_variant, variant_contract
        ),
        "executable_record_bytes": len(executable),
        "executable_record_sha256": _sha256(executable),
        "executable_record_sha256_is_path_dependent": True,
        "normalized_executable_record_bytes": len(normalized_executable),
        "normalized_executable_record_sha256": _sha256(normalized_executable),
        "normalized_hlo_module_bytes": len(normalized_hlo_module),
        "normalized_hlo_module_sha256": _sha256(normalized_hlo_module),
        "caller_bound_autotune_cache_path": os.fsdecode(autotune_cache_path),
        "caller_bound_autotune_cache_path_occurrences": 1,
        "caller_bound_autotune_cache_path_normalized": True,
        "caller_bound_autotune_cache_path_field_offset": (
            _EXPECTED_AUTOTUNE_PATH_FIELD_OFFSET
        ),
        "caller_bound_autotune_cache_path_record_offset": (
            _EXPECTED_AUTOTUNE_PATH_RECORD_OFFSET
        ),
        "thunk_count": len(inventory),
        "all_thunks_are_exact_custom_kernels": True,
        "sequential_wrapper_present": False,
        "device_to_device_copy_thunk_present": False,
        "ordered_thunks": inventory,
    }


def _cstring(table: bytes, offset: int, *, label: str) -> str:
    if offset < 0 or offset >= len(table):
        raise VerificationError(f"{label} string offset is invalid")
    search_end = min(len(table), offset + _MAX_ELF_NAME_BYTES + 1)
    end = table.find(b"\0", offset, search_end)
    if end < 0:
        if len(table) - offset > _MAX_ELF_NAME_BYTES:
            raise VerificationError(f"{label} string exceeds its length bound")
        raise VerificationError(f"{label} string is unterminated")
    try:
        return table[offset:end].decode("ascii")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} string is not ASCII") from error


def _parse_elf(
    data: bytes, offset: int, *, source: str
) -> tuple[bytes, dict[str, Any]]:
    if offset < 0 or len(data) - offset < 64:
        raise VerificationError(f"{source} ELF header is truncated")
    header = data[offset : offset + 64]
    if header[:7] != b"\x7fELF\x02\x01\x01" or header[7:9] != b"\x40\x03":
        raise VerificationError(f"{source} is not an ELF64 AMDGPU-HSA v3 object")
    (
        elf_type,
        machine,
        version,
        _entry,
        program_offset,
        section_offset,
        flags,
        header_size,
        program_entry_size,
        program_count,
        section_entry_size,
        section_count,
        section_name_index,
    ) = struct.unpack_from("<HHIQQQIHHHHHH", header, 16)
    if (
        elf_type != 3
        or machine != 0xE0
        or version != 1
        or flags != 0x41
        or header_size != 64
        or program_entry_size != 56
        or not 1 <= program_count <= 64
        or section_entry_size != 64
        or not 1 <= section_count <= 128
        or not 1 <= section_name_index < section_count
    ):
        raise VerificationError(f"{source} ELF header/target contract changed")
    relative_limit = len(data) - offset
    program_end = program_offset + program_entry_size * program_count
    section_end = section_offset + section_entry_size * section_count
    if max(program_end, section_end) > relative_limit:
        raise VerificationError(f"{source} ELF tables are truncated")
    file_end = max(64, program_end, section_end)
    for index in range(program_count):
        position = offset + program_offset + index * program_entry_size
        _, _, file_offset, _, _, file_size, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, position
        )
        if file_offset + file_size > relative_limit:
            raise VerificationError(f"{source} ELF program segment is truncated")
        file_end = max(file_end, file_offset + file_size)
    sections: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for index in range(section_count):
        position = offset + section_offset + index * section_entry_size
        section = struct.unpack_from("<IIQQQQIIQQ", data, position)
        sections.append(section)
        _, section_type, _, _, file_offset, file_size, _, _, _, _ = section
        if section_type != 8:
            if file_offset + file_size > relative_limit:
                raise VerificationError(f"{source} ELF section is truncated")
            file_end = max(file_end, file_offset + file_size)
    if file_end <= 0 or file_end > _MAX_ELF:
        raise VerificationError(f"{source} ELF size is outside the fixed bound")
    elf = data[offset : offset + file_end]
    name_section = sections[section_name_index]
    if name_section[1] != 3:
        raise VerificationError(
            f"{source} ELF section-name table is not a string table"
        )
    name_offset, name_size = name_section[4], name_section[5]
    names = elf[name_offset : name_offset + name_size]
    section_names = [_cstring(names, item[0], label="section") for item in sections]
    functions: set[str] = set()
    objects: set[str] = set()
    symbol_count = 0
    for section in sections:
        _, section_type, _, _, file_offset, file_size, link, _, _, entry_size = section
        if section_type not in (2, 11):
            continue
        if entry_size != 24 or file_size % entry_size or link >= len(sections):
            raise VerificationError(f"{source} ELF symbol table is malformed")
        table_symbol_count = file_size // entry_size
        if (
            table_symbol_count > _MAX_ELF_SYMBOLS
            or symbol_count + table_symbol_count > _MAX_ELF_SYMBOLS
        ):
            raise VerificationError(f"{source} ELF symbol count exceeds its bound")
        symbol_count += table_symbol_count
        linked = sections[link]
        if linked[1] != 3:
            raise VerificationError(f"{source} ELF symbols do not link a string table")
        string_table = elf[linked[4] : linked[4] + linked[5]]
        for entry_offset in range(file_offset, file_offset + file_size, entry_size):
            name, info, _other, shndx, _value, _size = struct.unpack_from(
                "<IBBHQQ", elf, entry_offset
            )
            if shndx != 0 and shndx >= len(sections):
                raise VerificationError(
                    f"{source} ELF symbol references an invalid or unsupported section"
                )
            symbol_name = _cstring(string_table, name, label="symbol")
            binding, symbol_type = info >> 4, info & 0xF
            if symbol_name and binding == 1 and shndx != 0:
                if symbol_type == 2:
                    functions.add(symbol_name)
                elif symbol_type == 1:
                    objects.add(symbol_name)
    return elf, {
        "source": source,
        "offset": offset,
        "bytes": len(elf),
        "sha256": _sha256(elf),
        "elf_format": "elf64-amdgpu",
        "machine": "EM_AMDGPU",
        "amdgpu_flags": flags,
        "target": "gfx1100",
        "program_header_count": program_count,
        "section_header_count": section_count,
        "section_names": section_names,
        "section_sizes": {
            name: sections[index][5] for index, name in enumerate(section_names) if name
        },
        "global_functions": sorted(functions),
        "global_objects": sorted(objects),
    }


def _inventory_elfs(
    records: list[bytes],
    *,
    expected_nested_elfs: Sequence[tuple[str, int, str]] | None = None,
) -> tuple[bytes, list[dict[str, Any]]]:
    if len(records) != 5 or records[1] or records[3]:
        raise VerificationError("split-proto record layout changed")
    inventory: list[dict[str, Any]] = []
    top, top_manifest = _parse_elf(records[2], 0, source="record[2]")
    module_identifier = records[2][len(top) :]
    if (
        len(top) != _EXPECTED_EMPTY_WRAPPER_ELF_BYTES
        or len(records[2]) != _EXPECTED_WRAPPER_RECORD_BYTES
        or len(module_identifier) != _EXPECTED_ROCM_MODULE_IDENTIFIER_BYTES
        or b"\x7fELF" in module_identifier
        or top_manifest["global_functions"]
        or top_manifest["global_objects"]
        or top_manifest["section_sizes"].get(".text") != 0
        or _sha256(top) != _EXPECTED_EMPTY_WRAPPER_ELF_SHA256
    ):
        raise VerificationError(
            "top-level executable is not the expected empty wrapper"
        )
    top_manifest["classification"] = "empty_top_level_wrapper_non_candidate"
    top_manifest["record_bytes"] = len(records[2])
    top_manifest["record_sha256"] = _sha256(records[2])
    top_manifest["trailing_bytes"] = len(module_identifier)
    top_manifest["trailing_sha256"] = _sha256(module_identifier)
    top_manifest["rocm_module_identifier"] = {
        "bytes": len(module_identifier),
        "sha256": _sha256(module_identifier),
        "timestamp_nanoseconds_little_endian": int.from_bytes(
            module_identifier[:8], "little"
        ),
        "random_identifier_u64_little_endian": int.from_bytes(
            module_identifier[8:], "little"
        ),
        "values_pinned": False,
        "integrity_bound_by_caller_cache_sha256": True,
    }
    inventory.append(top_manifest)
    nested = records[4]
    offsets: list[int] = []
    search_from = 0
    while True:
        elf_offset = nested.find(b"\x7fELF", search_from)
        if elf_offset < 0:
            break
        offsets.append(elf_offset)
        if len(offsets) > _MAX_NESTED_ELFS:
            raise VerificationError("nested ELF count is outside the fixed bound")
        search_from = elf_offset + 1
    if not offsets:
        raise VerificationError("nested ELF count is outside the fixed bound")
    previous_end = -1
    candidates: list[bytes] = []
    observed_nested: list[tuple[str, int, str]] = []
    for index, offset in enumerate(offsets):
        elf, manifest = _parse_elf(nested, offset, source=f"record[4].elf[{index}]")
        if offset < previous_end:
            raise VerificationError("nested ELF magic occurs inside another ELF")
        previous_end = offset + len(elf)
        inventory.append(manifest)
        functions = manifest["global_functions"]
        objects = manifest["global_objects"]
        if len(functions) != 1 or objects != [f"{functions[0]}.kd"]:
            raise VerificationError(
                "nested ELF function/descriptor is not paired or isolated as one kernel"
            )
        observed_nested.append((functions[0], len(elf), _sha256(elf)))
        defines_expected_function = _EXPECTED_SYMBOL in manifest["global_functions"]
        defines_expected_descriptor = (
            f"{_EXPECTED_SYMBOL}.kd" in manifest["global_objects"]
        )
        if defines_expected_function != defines_expected_descriptor:
            raise VerificationError(
                "expected function and descriptor symbols are not paired"
            )
        if defines_expected_function:
            if manifest["global_functions"] != [_EXPECTED_SYMBOL]:
                raise VerificationError(
                    "expected function is not isolated in one exact nested ELF"
                )
            if manifest["global_objects"] != [f"{_EXPECTED_SYMBOL}.kd"]:
                raise VerificationError("candidate kernel descriptor symbol changed")
            if offset < 2:
                raise VerificationError(
                    "candidate ELF is not bound by its exact protobuf field length"
                )
            prefix_start = offset - 1
            while prefix_start > 0 and nested[prefix_start - 1] >= 0x80:
                prefix_start -= 1
            if prefix_start == 0 or nested[prefix_start - 1] != 0x0A:
                raise VerificationError(
                    "candidate ELF is not bound by its exact protobuf field length"
                )
            encoded_size, prefix_end = _read_varint(
                nested, prefix_start, label="candidate protobuf field length"
            )
            if prefix_end != offset or encoded_size != len(elf):
                raise VerificationError(
                    "candidate ELF is not bound by its exact protobuf field length"
                )
            prefix = nested[prefix_start - 1 : offset]
            manifest["protobuf_field_prefix_hex"] = prefix.hex()
            candidates.append(elf)
    if len(candidates) != 1:
        raise VerificationError(
            "expected symbol does not identify one unique nested ELF"
        )
    expected_inventory = (
        _EXPECTED_NESTED_ELFS
        if expected_nested_elfs is None
        else tuple(expected_nested_elfs)
    )
    if tuple(observed_nested) != expected_inventory:
        raise VerificationError("ordered nested gfx1100 object inventory changed")
    candidate = candidates[0]
    if (
        len(candidate) != _EXPECTED_HSACO_BYTES
        or _sha256(candidate) != _EXPECTED_HSACO_SHA256
    ):
        raise VerificationError("candidate HSACO is not the fixed W8 microcase object")
    return candidate, inventory


def _exact_scalar(text: str, name: str) -> int:
    matches = re.findall(
        rf"^\s*\.{re.escape(name)}:\s+([0-9]+)\s*$", text, re.MULTILINE
    )
    if len(matches) != 1:
        raise VerificationError(f"readobj metadata field {name} is not unique")
    return int(matches[0])


def _inspect_tool_output(readobj: bytes, objdump: bytes) -> dict[str, Any]:
    try:
        readobj_text = readobj.decode("utf-8")
        objdump_text = objdump.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError("LLVM inspection output is not UTF-8") from error
    required_readobj = (
        "Format: elf64-amdgpu",
        "Arch: amdgcn",
        "EF_AMDGPU_MACH_AMDGCN_GFX1100 (0x41)",
        "Type: NT_AMDGPU_METADATA (AMDGPU Metadata)",
        "amdhsa.target:   amdgcn--amdhsa-amdgiz-gfx1100",
    )
    if any(readobj_text.count(marker) != 1 for marker in required_readobj):
        raise VerificationError("readobj target/metadata markers are not exact")
    if (
        len(
            re.findall(
                rf"^\s+Name: {re.escape(_EXPECTED_SYMBOL)} \(",
                readobj_text,
                re.MULTILINE,
            )
        )
        != 1
    ):
        raise VerificationError("readobj does not contain one exact function symbol")
    if (
        len(
            re.findall(
                rf"^\s+Name: {re.escape(_EXPECTED_SYMBOL)}\.kd \(",
                readobj_text,
                re.MULTILINE,
            )
        )
        != 1
    ):
        raise VerificationError("readobj does not contain one exact descriptor symbol")
    if (
        len(
            re.findall(
                rf"^\s*\.name:\s+{re.escape(_EXPECTED_SYMBOL)}\s*$",
                readobj_text,
                re.MULTILINE,
            )
        )
        != 1
    ):
        raise VerificationError("AMDGPU metadata kernel name is not exact")
    if (
        len(
            re.findall(
                rf"^\s*\.symbol:\s+{re.escape(_EXPECTED_SYMBOL)}\.kd\s*$",
                readobj_text,
                re.MULTILINE,
            )
        )
        != 1
    ):
        raise VerificationError("AMDGPU metadata descriptor is not exact")
    resources = {
        name: _exact_scalar(readobj_text, name) for name in _EXPECTED_RESOURCES
    }
    if resources != _EXPECTED_RESOURCES:
        raise VerificationError("AMDGPU resource metadata changed")
    if readobj_text.count(".uses_dynamic_stack: false") != 1:
        raise VerificationError("dynamic-stack metadata is not exact")
    argument_offsets = [
        int(value)
        for value in re.findall(
            r"^\s*\.offset:\s+([0-9]+)\s*$", readobj_text, re.MULTILINE
        )
    ]
    argument_sizes = [
        int(value)
        for value in re.findall(
            r"^\s*\.size:\s+([0-9]+)\s*$", readobj_text, re.MULTILINE
        )
    ]
    if argument_offsets != list(range(0, 64, 8)) or argument_sizes != [8] * 8:
        raise VerificationError("AMDGPU kernarg layout changed")
    if readobj_text.count(".value_kind:     global_buffer") != 8:
        raise VerificationError("AMDGPU kernarg kinds changed")
    if (
        readobj_text.count(".actual_access:  read_only") != 7
        or readobj_text.count(".actual_access:  write_only") != 1
    ):
        raise VerificationError("AMDGPU kernarg access metadata changed")
    labels = re.findall(r"^[0-9a-f]+ <([^>]+)>:$", objdump_text, re.MULTILINE)
    if labels != [_EXPECTED_SYMBOL]:
        raise VerificationError("objdump function boundary is not the exact kernel")
    canonical_instructions: list[str] = []
    raw_instruction_lines: list[str] = []
    for line in objdump_text.splitlines():
        if re.match(r"^\s*[a-z][a-z0-9_.]*\b", line) is None:
            continue
        canonical_instructions.append(line.split("//", 1)[0].strip())
        raw_instruction_lines.append(line)
    barrier_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if instruction == "s_barrier"
    ]
    branch_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if re.match(r"^s_(?:branch|cbranch_[a-z0-9_]+)\b", instruction)
    ]
    saveexec_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if instruction == "s_and_saveexec_b32 s0, vcc_lo"
    ]
    vcc_writer_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if re.match(r"^v_cmp_[a-z0-9_]+\s+vcc_lo,", instruction)
    ]
    end_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if instruction == "s_endpgm"
    ]
    deallocation_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if instruction == "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)"
    ]
    unsupported_control_transfer_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if re.match(
            r"^s_(?:setpc|swappc|call|rfe|trap|sethalt)(?:_[a-z0-9]+)?\b",
            instruction,
        )
    ]
    if (
        len(barrier_indices) != 9
        or len(branch_indices) != 1
        or canonical_instructions[branch_indices[0]] != "s_cbranch_execz 294"
        or len(saveexec_indices) != 1
        or len(
            pre_mask_vcc_writers := [
                index for index in vcc_writer_indices if index < saveexec_indices[0]
            ]
        )
        != 1
        or canonical_instructions[pre_mask_vcc_writers[0]]
        != "v_cmp_eq_u32_e32 vcc_lo, 0, v7"
        or len(end_indices) != 1
        or len(deallocation_indices) != 1
        or max(barrier_indices) >= saveexec_indices[0]
        or saveexec_indices[0] + 1 != branch_indices[0]
    ):
        raise VerificationError("full-tile barrier/EXEC/branch contract changed")
    if unsupported_control_transfer_indices:
        raise VerificationError("unsupported scalar control transfer is present")

    predicate_index = pre_mask_vcc_writers[0]
    saveexec_index = saveexec_indices[0]
    vcc_references_after_predicate = [
        index
        for index in range(predicate_index + 1, saveexec_index)
        if re.search(r"\bvcc(?:_lo|_hi)?\b", canonical_instructions[index])
    ]
    explicit_exec_references = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if re.search(r"\bexec(?:_lo|_hi)?\b", instruction)
    ]
    implicit_exec_writer_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if (opcode := instruction.split(None, 1)[0]).startswith("v_cmpx_")
        or "saveexec" in opcode
        or "wrexec" in opcode
    ]
    if (
        vcc_references_after_predicate
        or explicit_exec_references
        or implicit_exec_writer_indices != [saveexec_index]
    ):
        raise VerificationError(
            "full-tile predicate/EXEC state is explicitly modified outside the exact mask"
        )

    branch_raw = raw_instruction_lines[branch_indices[0]]
    branch_target = re.search(
        rf"<{re.escape(_EXPECTED_SYMBOL)}\+0x([0-9a-f]+)>", branch_raw
    )
    function_label = re.search(
        rf"^([0-9a-f]+) <{re.escape(_EXPECTED_SYMBOL)}>:$",
        objdump_text,
        re.MULTILINE,
    )
    if branch_target is None or function_label is None:
        raise VerificationError("full-tile branch target annotation changed")
    target_address = int(function_label.group(1), 16) + int(branch_target.group(1), 16)
    address_to_instruction: dict[int, tuple[int, str]] = {}
    for index, (raw, canonical_instruction) in enumerate(
        zip(raw_instruction_lines, canonical_instructions, strict=True)
    ):
        address = re.search(r"//\s*([0-9A-Fa-f]+):", raw)
        if address is not None:
            numeric_address = int(address.group(1), 16)
            if numeric_address in address_to_instruction:
                raise VerificationError("objdump instruction address is duplicated")
            address_to_instruction[numeric_address] = (index, canonical_instruction)
    branch_source = re.search(r"//\s*([0-9A-Fa-f]+):", branch_raw)
    if branch_source is None:
        raise VerificationError("full-tile branch source address is missing")
    branch_source_address = int(branch_source.group(1), 16)
    branch_immediate = int(canonical_instructions[branch_indices[0]].rsplit(" ", 1)[1])
    if not 0 <= branch_immediate <= 0xFFFF:
        raise VerificationError("full-tile branch immediate is outside signed16")
    signed_branch_immediate = (
        branch_immediate - 0x10000 if branch_immediate & 0x8000 else branch_immediate
    )
    computed_target_address = branch_source_address + 4 + signed_branch_immediate * 4
    if (
        address_to_instruction.get(branch_source_address)
        != (
            branch_indices[0],
            canonical_instructions[branch_indices[0]],
        )
        or computed_target_address != target_address
    ):
        raise VerificationError("full-tile branch source/target arithmetic changed")
    target = address_to_instruction.get(target_address)
    if (
        target is None
        or target[1] != "s_nop 0"
        or target[0] + 1 != deallocation_indices[0]
        or target[0] + 2 != end_indices[0]
        or branch_indices[0] >= target[0]
    ):
        raise VerificationError("full-tile branch does not reach the common epilogue")

    without_registers = re.sub(
        r"\b[vs]\[[^]]+\]|\b[vs][0-9]+\b", "", "\n".join(canonical_instructions)
    )
    literal_tail_count = len(
        re.findall(r"(?<![A-Za-z0-9_])(?:17|0x11)(?![A-Za-z0-9_])", without_registers)
    )
    ds_store_b8_count = sum(
        instruction.startswith("ds_store_b8 ") for instruction in canonical_instructions
    )
    global_store_indices = [
        index
        for index, instruction in enumerate(canonical_instructions)
        if instruction.startswith("global_store_")
    ]
    global_store_count = len(global_store_indices)
    if (
        literal_tail_count != 0
        or ds_store_b8_count != 0
        or global_store_count != 8
        or not all(
            branch_indices[0] < index < target[0] for index in global_store_indices
        )
    ):
        raise VerificationError("full-tile tail-store ISA contract changed")

    wmma_lines = [
        line.strip()
        for line in objdump_text.splitlines()
        if re.match(r"^\s*v_wmma_", line)
    ]
    exact_wmma = [
        line for line in wmma_lines if re.match(r"^v_wmma_i32_16x16x16_iu8\b", line)
    ]
    if len(wmma_lines) != 4 or len(exact_wmma) != 4:
        raise VerificationError(
            "candidate does not contain exactly four IU8 WMMA instructions"
        )
    if any("neg_lo:[1,1,0]" not in line for line in exact_wmma):
        raise VerificationError("IU8 WMMA neg_lo operand modifier changed")
    canonical = [line.split("//", 1)[0].rstrip() for line in exact_wmma]
    return {
        "symbol": _EXPECTED_SYMBOL,
        "amdgpu_target": "amdgcn--amdhsa-amdgiz-gfx1100",
        "resources": resources,
        "uses_dynamic_stack": False,
        "kernarg_count": 8,
        "kernarg_offsets": argument_offsets,
        "kernarg_sizes": argument_sizes,
        "read_only_kernargs": 7,
        "write_only_kernargs": 1,
        "instruction": "v_wmma_i32_16x16x16_iu8",
        "static_instruction_count": len(exact_wmma),
        "signed_neg_lo": [1, 1, 0],
        "canonical_instruction_lines": canonical,
        "canonical_instruction_sha256": _sha256(("\n".join(canonical) + "\n").encode()),
        "control_flow": {
            "barrier_count": len(barrier_indices),
            "all_barriers_before_exec_mask": True,
            "vcc_predicate": canonical_instructions[pre_mask_vcc_writers[0]],
            "exec_mask": canonical_instructions[saveexec_indices[0]],
            "scalar_branch_count": len(branch_indices),
            "scalar_branches": [
                canonical_instructions[index] for index in branch_indices
            ],
            "branch_direction": "forward_only",
            "branch_target_offset_hex": f"0x{int(branch_target.group(1), 16):x}",
            "branch_target_instruction": target[1],
            "branch_target_is_common_deallocation_epilogue": True,
            "barrier_after_exec_mask_count": 0,
            "backedge_count": 0,
            "endpgm_count": len(end_indices),
        },
        "tail_store": {
            "standalone_immediate_17_or_0x11_count": literal_tail_count,
            "ds_store_b8_count": ds_store_b8_count,
            "global_store_count": global_store_count,
        },
        "llvm_readobj_stdout_bytes": len(readobj),
        "llvm_readobj_stdout_sha256": _sha256(readobj),
        "llvm_objdump_stdout_bytes": len(objdump),
        "llvm_objdump_stdout_sha256": _sha256(objdump),
    }


def _make_sealed_memfd(data: bytes) -> int:
    if not hasattr(os, "memfd_create"):
        raise VerificationError(
            "memfd_create is required for private offline inspection"
        )
    flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
    descriptor = os.memfd_create("skyrl-w8a8-lora-hsaco", flags)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_private_exclusive(path: Path, data: bytes) -> dict[str, Any]:
    _path_has_no_symlink(path)
    parent = path.parent
    info = os.stat(parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise VerificationError(
            "ELF output parent must be a private directory owned by this uid"
        )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    created_info = os.fstat(descriptor)
    try:
        os.fchmod(descriptor, 0o600)
        mode_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(mode_info.st_mode)
            or mode_info.st_uid != os.getuid()
            or mode_info.st_nlink != 1
            or stat.S_IMODE(mode_info.st_mode) != 0o600
        ):
            raise VerificationError("created ELF output is not an exact private file")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            current = os.stat(path, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (
                created_info.st_dev,
                created_info.st_ino,
            ):
                os.unlink(path)
        finally:
            raise
    else:
        os.close(descriptor)
    return {
        "path": str(path),
        "bytes": len(data),
        "mode": stat.S_IMODE(mode_info.st_mode),
        "sha256": _sha256(data),
    }


def inspect_cache(
    cache_path: Path,
    *,
    extracted_elf_path: Path | None = None,
    expected_cache_sha256: str | None = None,
    expected_elf_sha256: str = _EXPECTED_HSACO_SHA256,
) -> dict[str, Any]:
    """Inspect one private cache entry without importing JAX or touching a device.

    ``expected_cache_sha256`` is typed as optional to make accidental omission
    produce a controlled ``VerificationError`` instead of a call-signature
    error.  It remains mandatory for a passing result.  The expected ELF digest
    is exposed for integration clarity but may not override this source
    revision's pinned microcase.
    """

    cache_path = Path(cache_path)
    if extracted_elf_path is not None:
        extracted_elf_path = Path(extracted_elf_path)
    jax_modules_before = {
        name
        for name in sys.modules
        if name == "jax" or name.startswith(("jax.", "jaxlib"))
    }
    if expected_cache_sha256 is None:
        raise VerificationError(
            "expected_cache_sha256 is required for fail-closed binding"
        )
    if _HEX_SHA256.fullmatch(expected_cache_sha256) is None:
        raise VerificationError("--expected-cache-sha256 must be lowercase hexadecimal")
    if expected_elf_sha256 != _EXPECTED_HSACO_SHA256:
        raise VerificationError(
            "expected_elf_sha256 may not override the pinned microcase"
        )
    toolchain = _attest_toolchain()
    cache_descriptor, before, cache_manifest = _open_private_input(cache_path)
    try:
        compressed = _read_fd(cache_descriptor, _MAX_COMPRESSED_CACHE, label="cache")
        observed_cache_sha256 = _sha256(compressed)
        if observed_cache_sha256 != expected_cache_sha256:
            raise VerificationError(
                "cache SHA-256 does not match the caller-bound digest"
            )
        os.lseek(cache_descriptor, 0, os.SEEK_SET)
        decompressed, _ = _run_bounded(
            [str(_ZSTDCAT), "--", f"/proc/self/fd/{cache_descriptor}"],
            maximum_stdout=_MAX_DECOMPRESSED_CACHE,
            timeout_seconds=15.0,
            pass_fds=(cache_descriptor,),
        )
        after = os.fstat(cache_descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(after, name) for name in identity_fields
        ):
            raise VerificationError("cache input changed during inspection")
    finally:
        os.close(cache_descriptor)
    cache_manifest["sha256"] = observed_cache_sha256
    cache_manifest["expected_sha256_matched"] = True
    records, serialization = _extract_cache_records(decompressed, _Snappy().decompress)
    thunk_inventory = _inspect_thunks(records, cache_path=cache_path)
    executable_variant = thunk_inventory.get("executable_variant")
    if not isinstance(executable_variant, str):
        raise VerificationError("inspector selected an invalid executable variant")
    variant_contract = _EXPECTED_EXECUTABLE_VARIANTS.get(executable_variant)
    if variant_contract is None:
        raise VerificationError("inspector selected an unknown executable variant")
    expected_thunks = variant_contract["thunks"]
    expected_nested_elfs = tuple(
        (thunk["kernel"], thunk["elf_bytes"], thunk["elf_sha256"])
        for thunk in expected_thunks
    )
    candidate, elf_inventory = _inventory_elfs(
        records, expected_nested_elfs=expected_nested_elfs
    )
    candidate_descriptor = _make_sealed_memfd(candidate)
    try:
        readobj, _ = _run_bounded(
            [
                str(_LLVM_READOBJ),
                "--file-header",
                "--sections",
                "--symbols",
                "--notes",
                f"/proc/self/fd/{candidate_descriptor}",
            ],
            maximum_stdout=_MAX_TOOL_STDOUT,
            timeout_seconds=30.0,
            pass_fds=(candidate_descriptor,),
        )
        objdump, _ = _run_bounded(
            [
                str(_LLVM_OBJDUMP),
                "--disassemble",
                "--mcpu=gfx1100",
                f"/proc/self/fd/{candidate_descriptor}",
            ],
            maximum_stdout=_MAX_TOOL_STDOUT,
            timeout_seconds=30.0,
            pass_fds=(candidate_descriptor,),
        )
    finally:
        os.close(candidate_descriptor)
    isa = _inspect_tool_output(readobj, objdump)
    jax_modules_after = {
        name
        for name in sys.modules
        if name == "jax" or name.startswith(("jax.", "jaxlib"))
    }
    if jax_modules_after != jax_modules_before:
        raise VerificationError("offline verifier caused a JAX module import")
    written = (
        _write_private_exclusive(extracted_elf_path, candidate)
        if extracted_elf_path
        else None
    )
    return {
        "schema": _SCHEMA,
        "status": "passed_offline_isa_verification",
        "executable_variant": executable_variant,
        "executable_variant_contract_sha256": thunk_inventory[
            "executable_variant_contract_sha256"
        ],
        "offline_only": True,
        "jax_modules_loaded_before": bool(jax_modules_before),
        "jax_modules_imported_by_verifier": False,
        "device_access_performed": False,
        "runtime_promotion": False,
        "cache": cache_manifest,
        "serialization": serialization,
        "thunk_inventory": thunk_inventory,
        "elf_inventory": {
            "executable_variant": executable_variant,
            "executable_variant_contract_sha256": thunk_inventory[
                "executable_variant_contract_sha256"
            ],
            "elf_count": len(elf_inventory),
            "nested_elf_count": len(elf_inventory) - 1,
            "unique_exact_symbol_candidate_count": sum(
                item["global_functions"] == [_EXPECTED_SYMBOL] for item in elf_inventory
            ),
            "ordered_nested_contract_matched": True,
            "files": elf_inventory,
        },
        "candidate": {
            "bytes": len(candidate),
            "sha256": _sha256(candidate),
            "expected_sha256_matched": True,
            "written_elf": written,
        },
        "isa": isa,
        "toolchain": toolchain,
        "claim": {
            "proves": "the caller-bound JAX cache contains the exact six-thunk full-tile executable, ordered gfx1100 objects, symbol-matched forward HSACO, four static IU8 WMMAs, and barrier-safe forward-only tail epilogue",
            "does_not_prove": [
                "Pallas provenance without paired exact HLO/custom-call evidence",
                "kernel dispatch",
                "runtime numerical correctness",
                "latency or throughput",
                "absence of runtime graph or capture APIs",
            ],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="private absolute JAX cache entry")
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument(
        "--elf-output",
        type=Path,
        help="optional new path; created private with O_EXCL and never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        evidence = inspect_cache(
            arguments.cache,
            expected_cache_sha256=arguments.expected_cache_sha256,
            extracted_elf_path=arguments.elf_output,
        )
    except (OSError, VerificationError) as error:
        print(
            json.dumps(
                {"schema": _SCHEMA, "status": "failed", "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
