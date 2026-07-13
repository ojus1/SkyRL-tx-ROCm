#!/usr/bin/python3
"""Fail-closed, offline ISA verifier for the fixed W8A8 LoRA compile probe.

This program understands only the JAX 0.10.2 cache shape emitted by
``probe_w8a8_lora_forward.py``.  It does not import JAX, enumerate a device, or
open a device node.  The external tool entry binaries and directly loaded
libsnappy object are SHA-256 pinned.  Their dynamic loaders and transitive
dependencies are not independently hash-closed.

The cache's top-level 1,920-byte wrapper record contains an empty 1,904-byte
ELF plus a 16-byte opaque trailer.  The generated code objects are nested in
the final split-proto record.  Qualification therefore requires one and only
one nested ELF with the exact forward symbol, the fixed gfx1100 resource
contract, and exactly four IU8 WMMA instructions with the expected ``neg_lo``
operand modifier.
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
_EXPECTED_HSACO_BYTES = 8_440
_EXPECTED_HSACO_SHA256 = (
    "606a80a508317af303966e5c2ca357d138d08828949c0dbfdcd73ccde1726389"
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
_EXPECTED_EMPTY_WRAPPER_RECORD_SHA256 = (
    "c2958461eb1d5ca0a040819d1bb365c3fa87e3e15e8bcc5730c2ec5dcf4ee180"
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
    "vgpr_count": 62,
    "vgpr_spill_count": 0,
    "wavefront_size": 32,
    "workgroup_processor_mode": 1,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    if sizes[:4] != [len(_EXPECTED_SPLIT_MANIFEST), 0, 1920, 0]:
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
        or section_name_index >= section_count
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


def _inventory_elfs(records: list[bytes]) -> tuple[bytes, list[dict[str, Any]]]:
    if len(records) != 5 or records[1] or records[3]:
        raise VerificationError("split-proto record layout changed")
    inventory: list[dict[str, Any]] = []
    top, top_manifest = _parse_elf(records[2], 0, source="record[2]")
    if (
        len(top) != 1904
        or len(records[2]) != 1920
        or top_manifest["global_functions"]
        or top_manifest["global_objects"]
        or top_manifest["section_sizes"].get(".text") != 0
        or _sha256(records[2]) != _EXPECTED_EMPTY_WRAPPER_RECORD_SHA256
    ):
        raise VerificationError(
            "top-level executable is not the expected empty wrapper"
        )
    top_manifest["classification"] = "empty_top_level_wrapper_non_candidate"
    top_manifest["record_bytes"] = len(records[2])
    top_manifest["record_sha256"] = _sha256(records[2])
    top_manifest["trailing_bytes"] = len(records[2]) - len(top)
    top_manifest["trailing_sha256"] = _sha256(records[2][len(top) :])
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
    for index, offset in enumerate(offsets):
        elf, manifest = _parse_elf(nested, offset, source=f"record[4].elf[{index}]")
        if offset < previous_end:
            raise VerificationError("nested ELF magic occurs inside another ELF")
        previous_end = offset + len(elf)
        inventory.append(manifest)
        if manifest["global_functions"] == [_EXPECTED_SYMBOL]:
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
    candidate = candidates[0]
    candidate_manifest = next(
        item for item in inventory if item["sha256"] == _sha256(candidate)
    )
    if candidate_manifest["global_objects"] != [f"{_EXPECTED_SYMBOL}.kd"]:
        raise VerificationError("candidate kernel descriptor symbol changed")
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
    candidate, elf_inventory = _inventory_elfs(records)
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
        "offline_only": True,
        "jax_modules_loaded_before": bool(jax_modules_before),
        "jax_modules_imported_by_verifier": False,
        "device_access_performed": False,
        "runtime_promotion": False,
        "cache": cache_manifest,
        "serialization": serialization,
        "elf_inventory": {
            "elf_count": len(elf_inventory),
            "unique_exact_symbol_candidate_count": sum(
                item["global_functions"] == [_EXPECTED_SYMBOL] for item in elf_inventory
            ),
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
            "proves": "the caller-bound JAX cache contains the exact symbol-matched gfx1100 HSACO with four static IU8 WMMA instructions and exact neg_lo operand modifiers",
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
