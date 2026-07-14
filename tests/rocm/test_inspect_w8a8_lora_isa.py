from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO / "rocm" / "inspect_w8a8_lora_isa.py"
_SPEC = importlib.util.spec_from_file_location("inspect_w8a8_lora_isa", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ISA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ISA)
_SYNTHETIC_CACHE_PATH = Path(
    "/tmp/skyrl-w8a8-compile-1234567890/compiler-artifacts/jax-cache/"
    f"jit_candidate-{'a' * 64}-cache"
)


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _align(value: int, alignment: int = 8) -> int:
    return (value + alignment - 1) // alignment * alignment


def _elf(
    function: str | None,
    *,
    flags: int = 0x41,
    text_size: int = 8,
    extent: int | None = None,
    extra_function: str | None = None,
    extra_object: str | None = None,
) -> bytes:
    section_names = b"\0.text\0.symtab\0.strtab\0.shstrtab\0.note\0"
    section_name_offsets = {
        name: section_names.index(name.encode())
        for name in (".text", ".symtab", ".strtab", ".shstrtab", ".note")
    }
    string_table = b"\0"
    function_offset = descriptor_offset = extra_function_offset = (
        extra_object_offset
    ) = 0
    if function is not None:
        function_offset = len(string_table)
        string_table += function.encode() + b"\0"
        descriptor_offset = len(string_table)
        string_table += f"{function}.kd".encode() + b"\0"
    if extra_function is not None:
        extra_function_offset = len(string_table)
        string_table += extra_function.encode() + b"\0"
    if extra_object is not None:
        extra_object_offset = len(string_table)
        string_table += extra_object.encode() + b"\0"
    symbols = bytearray(24)
    if function is not None:
        symbols += struct.pack(
            "<IBBHQQ", function_offset, 0x12, 3, 1, 0x1000, text_size
        )
        symbols += struct.pack("<IBBHQQ", descriptor_offset, 0x11, 0, 1, 0, 64)
    if extra_function is not None:
        symbols += struct.pack(
            "<IBBHQQ", extra_function_offset, 0x12, 3, 1, 0x1000, text_size
        )
    if extra_object is not None:
        symbols += struct.pack("<IBBHQQ", extra_object_offset, 0x11, 0, 1, 0, 64)

    cursor = 128
    text_offset = cursor
    cursor += text_size
    cursor = _align(cursor)
    string_offset = cursor
    cursor += len(string_table)
    cursor = _align(cursor)
    symbol_offset = cursor
    cursor += len(symbols)
    cursor = _align(cursor)
    section_names_offset = cursor
    cursor += len(section_names)
    cursor = _align(cursor)
    note_offset = cursor
    cursor += 1
    cursor = _align(cursor)
    section_count = 6
    minimum_extent = cursor + section_count * 64
    if extent is None:
        extent = minimum_extent
    if extent < minimum_extent:
        raise ValueError("synthetic ELF extent is too small")
    section_offset = extent - section_count * 64
    if section_offset < cursor:
        raise ValueError("synthetic section table overlaps data")

    data = bytearray(extent)
    ident = b"\x7fELF\x02\x01\x01\x40\x03" + b"\0" * 7
    data[:16] = ident
    struct.pack_into(
        "<HHIQQQIHHHHHH",
        data,
        16,
        3,
        0xE0,
        1,
        0,
        64,
        section_offset,
        flags,
        64,
        56,
        1,
        64,
        section_count,
        4,
    )
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0, 0, extent, extent, 0x1000)
    data[text_offset : text_offset + text_size] = b"\x90" * text_size
    data[string_offset : string_offset + len(string_table)] = string_table
    data[symbol_offset : symbol_offset + len(symbols)] = symbols
    data[section_names_offset : section_names_offset + len(section_names)] = (
        section_names
    )
    data[note_offset] = 0

    sections = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (
            section_name_offsets[".text"],
            1,
            6,
            0x1000,
            text_offset,
            text_size,
            0,
            0,
            16,
            0,
        ),
        (
            section_name_offsets[".symtab"],
            2,
            0,
            0,
            symbol_offset,
            len(symbols),
            3,
            1,
            8,
            24,
        ),
        (
            section_name_offsets[".strtab"],
            3,
            0,
            0,
            string_offset,
            len(string_table),
            0,
            0,
            1,
            0,
        ),
        (
            section_name_offsets[".shstrtab"],
            3,
            0,
            0,
            section_names_offset,
            len(section_names),
            0,
            0,
            1,
            0,
        ),
        (section_name_offsets[".note"], 7, 2, 0, note_offset, 1, 0, 0, 4, 0),
    ]
    for index, section in enumerate(sections):
        struct.pack_into("<IIQQQQIIQQ", data, section_offset + index * 64, *section)
    return bytes(data)


def _records(candidate: bytes) -> list[bytes]:
    wrapper = _elf(None, text_size=0, extent=1904) + b"opaque-trailer!!"
    assert len(wrapper) == 1920
    other = _elf("other_kernel")
    nested = b"proto-prefix" + b"\x0a" + _varint(len(other)) + other
    nested += b"between" + b"\x0a" + _varint(len(candidate)) + candidate + b"tail"
    return [_ISA._EXPECTED_SPLIT_MANIFEST, b"", wrapper, b"", nested]


def _bind_synthetic_wrapper(monkeypatch, records: list[bytes]) -> None:
    wrapper_elf = records[2][: -_ISA._EXPECTED_ROCM_MODULE_IDENTIFIER_BYTES]
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_EMPTY_WRAPPER_ELF_SHA256",
        hashlib.sha256(wrapper_elf).hexdigest(),
    )
    nested = records[4]
    offset = 0
    expected = []
    while True:
        offset = nested.find(b"\x7fELF", offset)
        if offset < 0:
            break
        elf, manifest = _ISA._parse_elf(nested, offset, source="synthetic")
        expected.append(
            (
                manifest["global_functions"][0],
                len(elf),
                hashlib.sha256(elf).hexdigest(),
            )
        )
        offset += len(elf)
    monkeypatch.setattr(_ISA, "_EXPECTED_NESTED_ELFS", tuple(expected))


def _split_proto(
    records: list[bytes], *, sizes_raw: bytes | None = None
) -> tuple[bytes, dict[bytes, bytes]]:
    sizes = (
        b"".join(_varint(len(record)) for record in records)
        if sizes_raw is None
        else sizes_raw
    )
    values = b"".join(records)
    encoded_sizes = _varint(len(sizes)) + b"S"
    encoded_values = _varint(len(values)) + b"V"
    chunk_data = b"s" + _varint(len(encoded_sizes)) + encoded_sizes + encoded_values
    header = bytearray(40)
    header[8:16] = len(chunk_data).to_bytes(8, "little")
    header[24] = ord("r")
    header[25:32] = len(records).to_bytes(7, "little")
    header[32:40] = len(values).to_bytes(8, "little")
    encoded = _ISA._RIEGELI_SIGNATURE + bytes(header) + chunk_data
    return encoded, {b"S": sizes, b"V": values}


def _decompressor(mapping: dict[bytes, bytes]):
    def decompress(encoded: bytes, expected: int) -> bytes:
        assert len(mapping[encoded]) == expected
        return mapping[encoded]

    return decompress


def _cache_payload(split_proto: bytes, metadata: bytes = b"M" * 292) -> bytes:
    executable = b"\x0a" + _varint(len(split_proto)) + split_proto
    executable += b"\x12" + _varint(3) + b"opt"
    return b"\0\0\0\0" + _varint(len(metadata)) + metadata + executable


def _bytes_field(number: int, value: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _varint_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _dim3d_field(wrapper_field: int, dimensions: tuple[int, int, int]) -> bytes:
    encoded = b"".join(
        _varint_field(index, value) for index, value in enumerate(dimensions, start=1)
    )
    return _bytes_field(wrapper_field, encoded)


def _synthetic_gpu_executable(
    *,
    grid_override: tuple[int, int, int] | None = None,
    non_custom_thunk_index: int | None = None,
    gemm_shared_memory_bytes: int | None = None,
    gemm_text_size: int = 8,
    cache_path: Path = _SYNTHETIC_CACHE_PATH,
    autotune_path_copies: int = 1,
    preexisting_normalized_path: bool = False,
    duplicate_directory_token: bool = False,
) -> tuple[bytes, tuple[dict[str, object], ...]]:
    thunks = []
    expected_thunks = []
    for index, source in enumerate(_ISA._EXPECTED_THUNKS):
        expected = dict(source)
        if index == 2 and gemm_shared_memory_bytes is not None:
            expected["shared_memory_bytes"] = gemm_shared_memory_bytes
        kernel_name = str(expected["kernel"])
        annotation = str(expected["annotation"])
        arguments = int(expected["arguments"])
        grid = tuple(expected["grid"])
        encoded_grid = grid_override if index == 4 and grid_override else grid
        threads = tuple(expected["threads"])
        elf = _elf(kernel_name, text_size=gemm_text_size if index == 2 else 8)
        loader = b"".join(
            (
                _bytes_field(2, _bytes_field(1, elf)),
                _varint_field(3, arguments),
                _bytes_field(4, kernel_name.encode()),
                _bytes_field(5, b"packing"),
            )
        )
        kernel = b"".join(
            (
                _bytes_field(1, kernel_name.encode()),
                _bytes_field(2, loader),
                _bytes_field(3, _dim3d_field(2, encoded_grid)),
                _bytes_field(4, _dim3d_field(1, threads)),
            )
        )
        shared = int(expected["shared_memory_bytes"])
        if shared:
            kernel += _varint_field(6, shared)
        custom = b"".join(_bytes_field(1, b"shaped-slice") for _ in range(arguments))
        custom += _bytes_field(2, bytes(expected["written"]))
        custom += _bytes_field(3, kernel) + _bytes_field(6, b"")
        info = _bytes_field(1, annotation.encode()) + _varint_field(
            3, int(expected["id"])
        )
        thunk_kind = 35 if index == non_custom_thunk_index else 36
        thunk = _bytes_field(1, info) + _bytes_field(thunk_kind, custom)
        expected.update(
            {
                "bytes": len(thunk),
                "sha256": hashlib.sha256(thunk).hexdigest(),
                "elf_bytes": len(elf),
                "elf_sha256": hashlib.sha256(elf).hexdigest(),
            }
        )
        thunks.append(thunk)
        expected_thunks.append(expected)
    autotune_cache_path = os.fsencode(
        cache_path.parent / os.fsdecode(_ISA._AUTOTUNE_CACHE_DIRECTORY)
    )
    hlo_module = b"fixed-prefix" + autotune_cache_path * autotune_path_copies
    if preexisting_normalized_path:
        hlo_module += _ISA._NORMALIZED_AUTOTUNE_CACHE_PATH
    if duplicate_directory_token:
        hlo_module += _ISA._AUTOTUNE_CACHE_DIRECTORY
    hlo_module += b"-fixed-suffix"
    executable = _bytes_field(1, hlo_module)
    executable += b"".join(
        _bytes_field(field, b"fixed") for field in (2, 6, 8, 9, 10, 11)
    )
    executable += b"".join(_bytes_field(13, thunk) for thunk in thunks)
    executable += _bytes_field(14, b"fixed")
    return executable, tuple(expected_thunks)


def _bind_synthetic_thunks(
    monkeypatch,
    executable: bytes,
    expected: tuple[dict[str, object], ...],
    *,
    cache_path: Path = _SYNTHETIC_CACHE_PATH,
) -> None:
    _bind_synthetic_variants(
        monkeypatch,
        {"synthetic": (executable, expected)},
        cache_path=cache_path,
    )


def _bind_synthetic_variants(
    monkeypatch,
    variants: dict[str, tuple[bytes, tuple[dict[str, object], ...]]],
    *,
    cache_path: Path = _SYNTHETIC_CACHE_PATH,
) -> None:
    autotune_cache_path = os.fsencode(
        cache_path.parent / os.fsdecode(_ISA._AUTOTUNE_CACHE_DIRECTORY)
    )
    contracts = {}
    first_executable = next(iter(variants.values()))[0]
    fields = _ISA._wire_fields(first_executable, label="synthetic", max_fields=32)
    hlo_module = fields[0][2]
    assert isinstance(hlo_module, bytes)
    for name, (executable, expected) in variants.items():
        normalized = executable.replace(
            autotune_cache_path, _ISA._NORMALIZED_AUTOTUNE_CACHE_PATH
        )
        normalized_fields = _ISA._wire_fields(
            normalized, label=f"normalized synthetic {name}", max_fields=32
        )
        normalized_hlo_module = normalized_fields[0][2]
        assert isinstance(normalized_hlo_module, bytes)
        contracts[name] = {
            "normalized_executable_record_bytes": len(normalized),
            "normalized_executable_record_sha256": hashlib.sha256(
                normalized
            ).hexdigest(),
            "normalized_hlo_module_bytes": len(normalized_hlo_module),
            "normalized_hlo_module_sha256": hashlib.sha256(
                normalized_hlo_module
            ).hexdigest(),
            "thunks": expected,
        }
    monkeypatch.setattr(_ISA, "_EXPECTED_EXECUTABLE_VARIANTS", contracts)
    monkeypatch.setattr(_ISA, "_EXPECTED_AUTOTUNE_PATH_BYTES", len(autotune_cache_path))
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_AUTOTUNE_PATH_FIELD_OFFSET",
        hlo_module.find(autotune_cache_path),
    )
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_AUTOTUNE_PATH_RECORD_OFFSET",
        first_executable.find(autotune_cache_path),
    )


def _readobj() -> bytes:
    resources = "\n".join(
        f"    .{name}: {value}" for name, value in _ISA._EXPECTED_RESOURCES.items()
    )
    arguments = []
    for index in range(8):
        access = "read_only" if index < 7 else "write_only"
        arguments.append(
            "\n".join(
                (
                    f"      - .actual_access:  {access}",
                    "        .address_space:  global",
                    f"        .name:           arg{index}",
                    f"        .offset:         {index * 8}",
                    "        .size:           8",
                    "        .value_kind:     global_buffer",
                )
            )
        )
    return (
        "Format: elf64-amdgpu\n"
        "Arch: amdgcn\n"
        "    EF_AMDGPU_MACH_AMDGCN_GFX1100 (0x41)\n"
        f"    Name: {_ISA._EXPECTED_SYMBOL} (1)\n"
        f"    Name: {_ISA._EXPECTED_SYMBOL}.kd (32)\n"
        "Type: NT_AMDGPU_METADATA (AMDGPU Metadata)\n"
        "amdhsa.kernels:\n"
        + "\n".join(arguments)
        + "\n"
        + resources
        + "\n"
        + f"    .name:           {_ISA._EXPECTED_SYMBOL}\n"
        + f"    .symbol:         {_ISA._EXPECTED_SYMBOL}.kd\n"
        + "    .uses_dynamic_stack: false\n"
        + "amdhsa.target:   amdgcn--amdhsa-amdgiz-gfx1100\n"
    ).encode()


def _objdump(count: int = 4, instruction: str = "v_wmma_i32_16x16x16_iu8") -> bytes:
    lines = [
        "/proc/self/fd/7:\tfile format elf64-amdgpu",
        "",
        f"0000000000001900 <{_ISA._EXPECTED_SYMBOL}>:",
    ]
    lines.extend(
        f"\t{instruction} v[1:8], v[10:13], v[14:17], v[1:8] neg_lo:[1,1,0]// {index:08X}"
        for index in range(count)
    )
    lines.append("\tv_cmp_eq_u32_e32 vcc_lo, 0, v7                 // 000000001AC0: 0")
    lines.extend(
        f"\ts_barrier                                             // {0x1B00 + 4 * index:016X}: 0"
        for index in range(9)
    )
    lines.extend(
        (
            "\ts_and_saveexec_b32 s0, vcc_lo                      // 00000000200C: 0",
            f"\ts_cbranch_execz 294                                  // 000000002010: 0 <{_ISA._EXPECTED_SYMBOL}+0xbac>",
        )
    )
    lines.extend(
        f"\tglobal_store_b16 v[0:1], v0, off                        // {0x2400 + 4 * index:016X}: 0"
        for index in range(8)
    )
    lines.extend(
        (
            "\ts_nop 0                                             // 0000000024AC: 0",
            "\ts_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                // 0000000024B0: 0",
            "\ts_endpgm                                            // 0000000024B4: 0",
        )
    )
    return ("\n".join(lines) + "\n").encode()


def test_parse_elf_finds_exact_global_symbols_and_target() -> None:
    elf = _elf(_ISA._EXPECTED_SYMBOL)

    extracted, manifest = _ISA._parse_elf(elf, 0, source="test")

    assert extracted == elf
    assert manifest["target"] == "gfx1100"
    assert manifest["global_functions"] == [_ISA._EXPECTED_SYMBOL]
    assert manifest["global_objects"] == [f"{_ISA._EXPECTED_SYMBOL}.kd"]
    assert manifest["section_sizes"][".text"] == 8


@pytest.mark.parametrize("mutation", ["target", "truncated", "magic", "embedded"])
def test_parse_elf_rejects_wrong_or_ambiguous_structure(mutation: str) -> None:
    elf = bytearray(_elf(_ISA._EXPECTED_SYMBOL))
    offset = 0
    if mutation == "target":
        struct.pack_into("<I", elf, 48, 0x40)
    elif mutation == "truncated":
        del elf[-1]
    elif mutation == "magic":
        elf[0] = 0
    else:
        elf[140:144] = b"\x7fELF"
        offset = 140

    with pytest.raises(_ISA.VerificationError):
        _ISA._parse_elf(bytes(elf), offset, source="test")


def test_parse_elf_rejects_symbol_table_object_amplification() -> None:
    elf = bytearray(_elf(_ISA._EXPECTED_SYMBOL, extent=8192))
    section_offset = struct.unpack_from("<Q", elf, 40)[0]
    symbol_size_offset = section_offset + 2 * 64 + 32
    struct.pack_into("<Q", elf, symbol_size_offset, (_ISA._MAX_ELF_SYMBOLS + 1) * 24)

    with pytest.raises(_ISA.VerificationError, match="symbol count"):
        _ISA._parse_elf(bytes(elf), 0, source="adversarial")


def test_parse_elf_rejects_non_string_section_name_table() -> None:
    elf = bytearray(_elf(_ISA._EXPECTED_SYMBOL))
    section_offset = struct.unpack_from("<Q", elf, 40)[0]
    section_name_index = struct.unpack_from("<H", elf, 62)[0]
    struct.pack_into("<I", elf, section_offset + section_name_index * 64 + 4, 1)

    with pytest.raises(_ISA.VerificationError, match="section-name table"):
        _ISA._parse_elf(bytes(elf), 0, source="adversarial")


def test_parse_elf_rejects_undefined_section_name_index() -> None:
    elf = bytearray(_elf(_ISA._EXPECTED_SYMBOL))
    section_offset = struct.unpack_from("<Q", elf, 40)[0]
    section_name_index = struct.unpack_from("<H", elf, 62)[0]
    source = section_offset + section_name_index * 64
    elf[section_offset : section_offset + 64] = elf[source : source + 64]
    struct.pack_into("<I", elf, section_offset, 0)
    struct.pack_into("<H", elf, 62, 0)

    with pytest.raises(_ISA.VerificationError, match="header/target contract"):
        _ISA._parse_elf(bytes(elf), 0, source="adversarial")


@pytest.mark.parametrize("section_name_index", [6, 0xFFFF])
def test_parse_elf_rejects_out_of_range_section_name_index(
    section_name_index: int,
) -> None:
    elf = bytearray(_elf(_ISA._EXPECTED_SYMBOL))
    struct.pack_into("<H", elf, 62, section_name_index)

    with pytest.raises(_ISA.VerificationError, match="header/target contract"):
        _ISA._parse_elf(bytes(elf), 0, source="adversarial")


@pytest.mark.parametrize("section_index", [6, 0xFF00, 0xFFF1, 0xFFF2, 0xFFFF])
def test_parse_elf_rejects_symbol_section_index_outside_fixed_contract(
    section_index: int,
) -> None:
    elf = bytearray(_elf(_ISA._EXPECTED_SYMBOL))
    section_offset = struct.unpack_from("<Q", elf, 40)[0]
    symbol_offset = struct.unpack_from("<Q", elf, section_offset + 2 * 64 + 24)[0]
    first_defined_symbol_section_index = symbol_offset + 24 + 6
    struct.pack_into("<H", elf, first_defined_symbol_section_index, section_index)

    with pytest.raises(_ISA.VerificationError, match="invalid or unsupported"):
        _ISA._parse_elf(bytes(elf), 0, source="adversarial")


def test_cstring_rejects_name_allocation_amplification() -> None:
    adversarial = b"A" * (_ISA._MAX_ELF_NAME_BYTES + 1) + b"\0"

    with pytest.raises(_ISA.VerificationError, match="length bound"):
        _ISA._cstring(adversarial, 0, label="adversarial")


def test_wire_field_decoder_stops_before_third_field(monkeypatch) -> None:
    original = _ISA._read_varint
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_ISA, "_read_varint", counted)
    adversarial = b"\x08\x00" * 3

    with pytest.raises(_ISA.VerificationError, match="field-count bound"):
        _ISA._wire_fields(adversarial, label="adversarial", max_fields=2)

    assert calls == 4


def test_split_proto_decoder_binds_exact_manifest_and_record_layout() -> None:
    records = _records(_elf(_ISA._EXPECTED_SYMBOL))
    encoded, mapping = _split_proto(records)

    actual, evidence = _ISA._extract_split_records(encoded, _decompressor(mapping))

    assert actual == records
    assert evidence["record_sizes"] == [56, 0, 1920, 0, len(records[4])]
    assert evidence["manifest_sha256"] == _ISA._EXPECTED_SPLIT_MANIFEST_SHA256
    assert evidence["riegeli_internal_hashes_validated"] is False


@pytest.mark.parametrize(
    "mutation", ["compression", "manifest", "record_count", "duplicate_signature"]
)
def test_split_proto_decoder_rejects_format_decoys(mutation: str) -> None:
    records = _records(_elf(_ISA._EXPECTED_SYMBOL))
    encoded, mapping = _split_proto(records)
    if mutation == "compression":
        encoded = encoded[:104] + b"z" + encoded[105:]
    elif mutation == "manifest":
        values = bytearray(mapping[b"V"])
        values[0] ^= 1
        mapping[b"V"] = bytes(values)
    elif mutation == "record_count":
        encoded = encoded[:89] + (6).to_bytes(7, "little") + encoded[96:]
    else:
        encoded += _ISA._RIEGELI_SIGNATURE

    with pytest.raises(_ISA.VerificationError):
        _ISA._extract_split_records(encoded, _decompressor(mapping))


def test_split_proto_decoder_rejects_record_size_object_amplification() -> None:
    records = _records(_elf(_ISA._EXPECTED_SYMBOL))
    exact_sizes = b"".join(_varint(len(record)) for record in records)
    mutations = (
        (b"\x00" * (_ISA._MAX_RECORD_SIZES_BYTES + 1), "decoded size"),
        (exact_sizes + b"\x00", "record count"),
    )

    for sizes_raw, message in mutations:
        encoded, mapping = _split_proto(records, sizes_raw=sizes_raw)
        with pytest.raises(_ISA.VerificationError, match=message):
            _ISA._extract_split_records(encoded, _decompressor(mapping))


def test_cache_decoder_requires_exact_ifrt_metadata_and_field_order(
    monkeypatch,
) -> None:
    records = _records(_elf(_ISA._EXPECTED_SYMBOL))
    split_proto, mapping = _split_proto(records)
    metadata = b"M" * 292
    monkeypatch.setattr(
        _ISA, "_EXPECTED_METADATA_SHA256", hashlib.sha256(metadata).hexdigest()
    )

    actual, evidence = _ISA._extract_cache_records(
        _cache_payload(split_proto, metadata), _decompressor(mapping)
    )

    assert actual == records
    assert evidence["ifrt_metadata_bytes"] == 292
    assert evidence["compile_options_bytes"] == 3

    reordered = b"\0\0\0\0" + _varint(292) + metadata
    reordered += b"\x12\x03opt\x0a" + _varint(len(split_proto)) + split_proto
    with pytest.raises(_ISA.VerificationError, match="exactly"):
        _ISA._extract_cache_records(reordered, _decompressor(mapping))


def test_thunk_decoder_pins_six_ordered_custom_kernel_launches(monkeypatch) -> None:
    executable, expected = _synthetic_gpu_executable()
    _bind_synthetic_thunks(monkeypatch, executable, expected)
    records = [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", executable]

    evidence = _ISA._inspect_thunks(records, cache_path=_SYNTHETIC_CACHE_PATH)

    assert evidence["thunk_count"] == 6
    assert evidence["all_thunks_are_exact_custom_kernels"] is True
    assert evidence["sequential_wrapper_present"] is False
    assert evidence["device_to_device_copy_thunk_present"] is False
    assert evidence["executable_variant"] == "synthetic"
    contract = _ISA._EXPECTED_EXECUTABLE_VARIANTS["synthetic"]
    assert evidence["executable_variant_contract_sha256"] == (
        _ISA._variant_contract_sha256("synthetic", contract)
    )
    assert evidence["executable_record_sha256_is_path_dependent"] is True
    assert evidence["caller_bound_autotune_cache_path_normalized"] is True
    assert evidence["caller_bound_autotune_cache_path_occurrences"] == 1
    assert evidence["caller_bound_autotune_cache_path_field_offset"] == (
        _ISA._EXPECTED_AUTOTUNE_PATH_FIELD_OFFSET
    )
    assert evidence["caller_bound_autotune_cache_path_record_offset"] == (
        _ISA._EXPECTED_AUTOTUNE_PATH_RECORD_OFFSET
    )
    assert (
        evidence["normalized_hlo_module_bytes"]
        == contract["normalized_hlo_module_bytes"]
    )
    assert (
        evidence["normalized_hlo_module_sha256"]
        == contract["normalized_hlo_module_sha256"]
    )
    ordered = evidence["ordered_thunks"]
    assert [item["thunk_id"] for item in ordered] == [1, 2, 3, 4, 5, 6]
    assert ordered[4]["kernel"] == _ISA._EXPECTED_SYMBOL
    assert ordered[4]["grid"] == [1, 2, 1]
    assert ordered[4]["threads"] == [128, 1, 1]
    assert ordered[4]["shared_memory_bytes"] == 1024
    assert ordered[5]["kernel"] == "wrapped_slice"
    assert ordered[5]["threads"] == [51, 1, 1]


def test_production_executable_variants_pin_both_complete_contracts() -> None:
    bm16 = _ISA._EXPECTED_EXECUTABLE_VARIANTS["lora_gemm_bm16_bn16"]
    bm32 = _ISA._EXPECTED_EXECUTABLE_VARIANTS["lora_gemm_bm32_bn32"]

    assert (
        bm16["normalized_executable_record_sha256"]
        == "94e1a986416c6b1b0b3d249b5ff41c2fc11dec215612a66c21d28a15968d49bc"
    )
    assert (
        bm32["normalized_executable_record_sha256"]
        == "989798f1183a243fe074491578827e4b04bf2d0eb25ca127f0a3b06f93050b94"
    )
    assert bm16["thunks"][2]["shared_memory_bytes"] == 8_192
    assert bm16["thunks"][2]["elf_sha256"].startswith("c45a0fb7")
    assert bm32["thunks"][2]["shared_memory_bytes"] == 16_384
    assert bm32["thunks"][2]["elf_sha256"].startswith("9ab0e3ab")
    assert _ISA._variant_contract_sha256("lora_gemm_bm16_bn16", bm16) == (
        "d2859c3fd661ca42f7ce2231d3b090af59e53210fad860519df7695a7856a947"
    )
    assert _ISA._variant_contract_sha256("lora_gemm_bm32_bn32", bm32) == (
        "749fff3d982c91f738c7b7c5c44d4d7120c9d24f439d10df90f20ae7a5890766"
    )


def test_thunk_decoder_accepts_both_atomic_executable_variants(monkeypatch) -> None:
    bm16 = _synthetic_gpu_executable()
    bm32 = _synthetic_gpu_executable(gemm_shared_memory_bytes=16_384, gemm_text_size=16)
    variants = {"bm16": bm16, "bm32": bm32}
    _bind_synthetic_variants(monkeypatch, variants)

    for name, (executable, _expected) in variants.items():
        records = [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", executable]
        evidence = _ISA._inspect_thunks(records, cache_path=_SYNTHETIC_CACHE_PATH)
        assert evidence["executable_variant"] == name
        assert evidence["executable_variant_contract_sha256"] == (
            _ISA._variant_contract_sha256(
                name, _ISA._EXPECTED_EXECUTABLE_VARIANTS[name]
            )
        )
    assert (
        _ISA._EXPECTED_EXECUTABLE_VARIANTS["bm16"]["thunks"][2]["shared_memory_bytes"]
        == 8_192
    )
    assert (
        _ISA._EXPECTED_EXECUTABLE_VARIANTS["bm32"]["thunks"][2]["shared_memory_bytes"]
        == 16_384
    )


def test_thunk_decoder_rejects_unknown_and_mixed_executable_variants(
    monkeypatch,
) -> None:
    bm16_executable, bm16_expected = _synthetic_gpu_executable()
    bm32_executable, bm32_expected = _synthetic_gpu_executable(
        gemm_shared_memory_bytes=16_384, gemm_text_size=16
    )
    records = [
        _ISA._EXPECTED_SPLIT_MANIFEST,
        b"",
        b"wrapper",
        b"",
        bm32_executable,
    ]

    _bind_synthetic_variants(monkeypatch, {"bm16": (bm16_executable, bm16_expected)})
    with pytest.raises(_ISA.VerificationError, match="one exact full-tile variant"):
        _ISA._inspect_thunks(records, cache_path=_SYNTHETIC_CACHE_PATH)

    _bind_synthetic_variants(monkeypatch, {"mixed": (bm32_executable, bm16_expected)})
    with pytest.raises(_ISA.VerificationError, match="serialization changed"):
        _ISA._inspect_thunks(records, cache_path=_SYNTHETIC_CACHE_PATH)

    assert bm32_expected != bm16_expected


def test_thunk_decoder_rejects_variant_with_altered_lds_or_elf(monkeypatch) -> None:
    lds_executable, lds_expected = _synthetic_gpu_executable(
        gemm_shared_memory_bytes=16_385, gemm_text_size=16
    )
    lds_contract = tuple(dict(item) for item in lds_expected)
    lds_contract[2]["shared_memory_bytes"] = 16_384
    _bind_synthetic_variants(
        monkeypatch, {"altered_lds": (lds_executable, lds_contract)}
    )
    with pytest.raises(_ISA.VerificationError, match="LDS allocation changed"):
        _ISA._inspect_thunks(
            [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", lds_executable],
            cache_path=_SYNTHETIC_CACHE_PATH,
        )

    elf_executable, elf_expected = _synthetic_gpu_executable(
        gemm_shared_memory_bytes=16_384, gemm_text_size=24
    )
    reference_executable, reference_expected = _synthetic_gpu_executable(
        gemm_shared_memory_bytes=16_384, gemm_text_size=16
    )
    del reference_executable
    elf_contract = tuple(dict(item) for item in elf_expected)
    elf_contract[2]["elf_bytes"] = reference_expected[2]["elf_bytes"]
    elf_contract[2]["elf_sha256"] = reference_expected[2]["elf_sha256"]
    _bind_synthetic_variants(
        monkeypatch, {"altered_elf": (elf_executable, elf_contract)}
    )
    with pytest.raises(_ISA.VerificationError, match="embedded gfx1100 object changed"):
        _ISA._inspect_thunks(
            [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", elf_executable],
            cache_path=_SYNTHETIC_CACHE_PATH,
        )


def test_thunk_decoder_normalizes_only_the_caller_bound_run_path(monkeypatch) -> None:
    runtime_cache = Path(
        "/tmp/skyrl-w8a8-runtime-9876543210/compiler-artifacts/jax-cache/"
        f"jit_candidate-{'b' * 64}-cache"
    )
    compile_executable, expected = _synthetic_gpu_executable()
    runtime_executable, _ = _synthetic_gpu_executable(cache_path=runtime_cache)
    _bind_synthetic_thunks(monkeypatch, compile_executable, expected)
    compile_autotune_path = os.fsencode(
        _SYNTHETIC_CACHE_PATH.parent / os.fsdecode(_ISA._AUTOTUNE_CACHE_DIRECTORY)
    )
    assert len(compile_autotune_path) == _ISA._EXPECTED_AUTOTUNE_PATH_BYTES
    assert len(_ISA._NORMALIZED_AUTOTUNE_CACHE_PATH) == len(compile_autotune_path)

    compile_evidence = _ISA._inspect_thunks(
        [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", compile_executable],
        cache_path=_SYNTHETIC_CACHE_PATH,
    )
    runtime_evidence = _ISA._inspect_thunks(
        [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", runtime_executable],
        cache_path=runtime_cache,
    )

    assert (
        compile_evidence["executable_record_sha256"]
        != runtime_evidence["executable_record_sha256"]
    )
    assert (
        compile_evidence["normalized_executable_record_sha256"]
        == runtime_evidence["normalized_executable_record_sha256"]
    )


def test_thunk_decoder_rejects_path_mismatch_duplicate_and_noncanonical_layout(
    monkeypatch,
) -> None:
    executable, expected = _synthetic_gpu_executable()
    _bind_synthetic_thunks(monkeypatch, executable, expected)
    records = [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", executable]
    mismatched_cache = Path(
        "/tmp/skyrl-w8a8-compile-1234567891/compiler-artifacts/jax-cache/"
        f"jit_candidate-{'a' * 64}-cache"
    )

    with pytest.raises(_ISA.VerificationError, match="uniquely caller-bound"):
        _ISA._inspect_thunks(records, cache_path=mismatched_cache)

    noncanonical_cache = Path(
        "/tmp/arbitrary-run-1234567890/compiler-artifacts/jax-cache/"
        f"jit_candidate-{'a' * 64}-cache"
    )
    noncanonical, noncanonical_expected = _synthetic_gpu_executable(
        cache_path=noncanonical_cache
    )
    _bind_synthetic_thunks(
        monkeypatch,
        noncanonical,
        noncanonical_expected,
        cache_path=noncanonical_cache,
    )
    with pytest.raises(_ISA.VerificationError, match="fresh /tmp run-scoped"):
        _ISA._inspect_thunks(
            [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", noncanonical],
            cache_path=noncanonical_cache,
        )

    wrong_basename = _SYNTHETIC_CACHE_PATH.with_name("arbitrary-cache-entry")
    with pytest.raises(_ISA.VerificationError, match="fresh /tmp run-scoped"):
        _ISA._inspect_thunks(records, cache_path=wrong_basename)

    duplicate, duplicate_expected = _synthetic_gpu_executable(autotune_path_copies=2)
    _bind_synthetic_thunks(monkeypatch, duplicate, duplicate_expected)
    with pytest.raises(_ISA.VerificationError, match="uniquely caller-bound"):
        _ISA._inspect_thunks(
            [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", duplicate],
            cache_path=_SYNTHETIC_CACHE_PATH,
        )


def test_thunk_decoder_rejects_moved_or_preexisting_normalized_path(
    monkeypatch,
) -> None:
    executable, expected = _synthetic_gpu_executable()
    _bind_synthetic_thunks(monkeypatch, executable, expected)
    fields = _ISA._wire_fields(executable, label="synthetic", max_fields=32)
    hlo_module = fields[0][2]
    assert isinstance(hlo_module, bytes)
    encoded_hlo = _bytes_field(1, hlo_module)
    moved = executable.replace(encoded_hlo, _bytes_field(1, b"x" + hlo_module[:-1]), 1)
    with pytest.raises(_ISA.VerificationError, match="uniquely caller-bound"):
        _ISA._inspect_thunks(
            [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", moved],
            cache_path=_SYNTHETIC_CACHE_PATH,
        )

    preexisting, preexisting_expected = _synthetic_gpu_executable(
        preexisting_normalized_path=True
    )
    _bind_synthetic_thunks(monkeypatch, preexisting, preexisting_expected)
    with pytest.raises(_ISA.VerificationError, match="uniquely caller-bound"):
        _ISA._inspect_thunks(
            [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", preexisting],
            cache_path=_SYNTHETIC_CACHE_PATH,
        )

    duplicate_token, duplicate_token_expected = _synthetic_gpu_executable(
        duplicate_directory_token=True
    )
    _bind_synthetic_thunks(monkeypatch, duplicate_token, duplicate_token_expected)
    with pytest.raises(_ISA.VerificationError, match="uniquely caller-bound"):
        _ISA._inspect_thunks(
            [
                _ISA._EXPECTED_SPLIT_MANIFEST,
                b"",
                b"wrapper",
                b"",
                duplicate_token,
            ],
            cache_path=_SYNTHETIC_CACHE_PATH,
        )


def test_thunk_decoder_rejects_nonpath_record_mutation(monkeypatch) -> None:
    executable, expected = _synthetic_gpu_executable()
    _bind_synthetic_thunks(monkeypatch, executable, expected)
    mutated = executable.replace(b"fixed", b"fIxed", 1)

    with pytest.raises(_ISA.VerificationError, match="normalized GPU executable"):
        _ISA._inspect_thunks(
            [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", mutated],
            cache_path=_SYNTHETIC_CACHE_PATH,
        )


def test_thunk_decoder_rejects_changed_pallas_grid_after_exact_raw_rebind(
    monkeypatch,
) -> None:
    executable, expected = _synthetic_gpu_executable(grid_override=(1, 3, 1))
    _bind_synthetic_thunks(monkeypatch, executable, expected)
    records = [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", executable]

    with pytest.raises(_ISA.VerificationError, match="launch dimensions"):
        _ISA._inspect_thunks(records, cache_path=_SYNTHETIC_CACHE_PATH)


def test_thunk_decoder_rejects_a_seventh_thunk(monkeypatch) -> None:
    executable, expected = _synthetic_gpu_executable()
    executable += _bytes_field(13, _bytes_field(1, b"copy"))
    _bind_synthetic_thunks(monkeypatch, executable, expected)
    records = [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", executable]

    with pytest.raises(_ISA.VerificationError, match="field sequence"):
        _ISA._inspect_thunks(records, cache_path=_SYNTHETIC_CACHE_PATH)


def test_thunk_decoder_rejects_non_custom_kind_with_six_thunks(monkeypatch) -> None:
    executable, expected = _synthetic_gpu_executable(non_custom_thunk_index=2)
    _bind_synthetic_thunks(monkeypatch, executable, expected)
    records = [_ISA._EXPECTED_SPLIT_MANIFEST, b"", b"wrapper", b"", executable]

    with pytest.raises(_ISA.VerificationError, match="not the exact custom-kernel"):
        _ISA._inspect_thunks(records, cache_path=_SYNTHETIC_CACHE_PATH)


def test_inventory_selects_symbol_not_position_and_classifies_empty_wrapper(
    monkeypatch,
) -> None:
    candidate = _elf(_ISA._EXPECTED_SYMBOL)
    records = _records(candidate)
    _bind_synthetic_wrapper(monkeypatch, records)
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_BYTES", len(candidate))
    monkeypatch.setattr(
        _ISA, "_EXPECTED_HSACO_SHA256", hashlib.sha256(candidate).hexdigest()
    )

    actual, inventory = _ISA._inventory_elfs(records)

    assert actual == candidate
    assert len(inventory) == 3
    assert inventory[0]["classification"] == "empty_top_level_wrapper_non_candidate"
    assert inventory[0]["global_functions"] == []
    assert inventory[0]["rocm_module_identifier"]["bytes"] == 16
    assert inventory[0]["rocm_module_identifier"]["values_pinned"] is False
    assert inventory[0]["trailing_bytes"] == 16
    assert (
        inventory[0]["trailing_sha256"]
        == inventory[0]["rocm_module_identifier"]["sha256"]
    )
    selected = [
        item
        for item in inventory
        if item["global_functions"] == [_ISA._EXPECTED_SYMBOL]
    ]
    assert len(selected) == 1
    assert selected[0]["protobuf_field_prefix_hex"].startswith("0a")


def test_inventory_rejects_duplicate_symbol_and_unbound_magic(monkeypatch) -> None:
    candidate = _elf(_ISA._EXPECTED_SYMBOL)
    records = _records(candidate)
    _bind_synthetic_wrapper(monkeypatch, records)
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_BYTES", len(candidate))
    monkeypatch.setattr(
        _ISA, "_EXPECTED_HSACO_SHA256", hashlib.sha256(candidate).hexdigest()
    )

    duplicate = list(records)
    duplicate[4] += b"\x0a" + _varint(len(candidate)) + candidate
    with pytest.raises(_ISA.VerificationError, match="unique"):
        _ISA._inventory_elfs(duplicate)

    unbound = list(records)
    candidate_offset = unbound[4].rfind(candidate)
    unbound[4] = (
        unbound[4][: candidate_offset - 2] + b"xx" + unbound[4][candidate_offset:]
    )
    with pytest.raises(_ISA.VerificationError, match="protobuf"):
        _ISA._inventory_elfs(unbound)

    decoy = _elf(_ISA._EXPECTED_SYMBOL, extra_function="other_kernel")
    decoy_nested = list(records)
    decoy_nested[4] += b"\x0a" + _varint(len(decoy)) + decoy
    with pytest.raises(_ISA.VerificationError, match="isolated"):
        _ISA._inventory_elfs(decoy_nested)

    extra_object = _elf(_ISA._EXPECTED_SYMBOL, extra_object="other_object")
    extra_object_nested = list(records)
    extra_object_nested[4] += b"\x0a" + _varint(len(extra_object)) + extra_object
    with pytest.raises(_ISA.VerificationError, match="descriptor"):
        _ISA._inventory_elfs(extra_object_nested)

    function_only = bytearray(_elf(_ISA._EXPECTED_SYMBOL))
    descriptor = f"{_ISA._EXPECTED_SYMBOL}.kd".encode()
    descriptor_offset = function_only.find(descriptor)
    assert descriptor_offset >= 0
    function_only[descriptor_offset : descriptor_offset + len(descriptor)] = (
        b"x" * len(_ISA._EXPECTED_SYMBOL) + b".kd"
    )
    function_only_nested = list(records)
    function_only_nested[4] += (
        b"\x0a" + _varint(len(function_only)) + bytes(function_only)
    )
    with pytest.raises(_ISA.VerificationError, match="not paired"):
        _ISA._inventory_elfs(function_only_nested)

    descriptor_only = bytearray(_elf(_ISA._EXPECTED_SYMBOL))
    function = _ISA._EXPECTED_SYMBOL.encode()
    function_offset = descriptor_only.find(function)
    assert function_offset >= 0
    descriptor_only[function_offset : function_offset + len(function)] = b"x" * len(
        function
    )
    descriptor_only_nested = list(records)
    descriptor_only_nested[4] += (
        b"\x0a" + _varint(len(descriptor_only)) + bytes(descriptor_only)
    )
    with pytest.raises(_ISA.VerificationError, match="not paired"):
        _ISA._inventory_elfs(descriptor_only_nested)


def test_inventory_accepts_variable_rocm_identifier_and_rejects_wrapper_mutations(
    monkeypatch,
) -> None:
    candidate = _elf(_ISA._EXPECTED_SYMBOL)
    records = _records(candidate)
    _bind_synthetic_wrapper(monkeypatch, records)
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_BYTES", len(candidate))
    monkeypatch.setattr(
        _ISA, "_EXPECTED_HSACO_SHA256", hashlib.sha256(candidate).hexdigest()
    )

    varied = list(records)
    varied[2] = varied[2][:-16] + bytes(range(16))
    _candidate, inventory = _ISA._inventory_elfs(varied)
    identifier = inventory[0]["rocm_module_identifier"]
    assert identifier["timestamp_nanoseconds_little_endian"] == int.from_bytes(
        bytes(range(8)), "little"
    )
    assert identifier["random_identifier_u64_little_endian"] == int.from_bytes(
        bytes(range(8, 16)), "little"
    )

    mutated_elf = list(records)
    wrapper = bytearray(mutated_elf[2])
    wrapper[1_000] ^= 1
    mutated_elf[2] = bytes(wrapper)
    with pytest.raises(_ISA.VerificationError, match="expected empty wrapper"):
        _ISA._inventory_elfs(mutated_elf)

    embedded_elf = list(records)
    embedded_elf[2] = embedded_elf[2][:-16] + b"\x7fELF" + b"x" * 12
    with pytest.raises(_ISA.VerificationError, match="expected empty wrapper"):
        _ISA._inventory_elfs(embedded_elf)

    short_identifier = list(records)
    short_identifier[2] = short_identifier[2][:-1]
    with pytest.raises(_ISA.VerificationError, match="expected empty wrapper"):
        _ISA._inventory_elfs(short_identifier)


def test_inventory_rejects_nested_elf_magic_amplification(monkeypatch) -> None:
    records = _records(_elf(_ISA._EXPECTED_SYMBOL))
    original = _ISA._parse_elf
    parsed_sources = []

    def counted(data, offset, *, source):
        parsed_sources.append(source)
        return original(data, offset, source=source)

    monkeypatch.setattr(_ISA, "_parse_elf", counted)
    _bind_synthetic_wrapper(monkeypatch, records)
    parsed_sources.clear()
    records[4] = b"prefix" + b"\x7fELF" * 33

    with pytest.raises(_ISA.VerificationError, match="nested ELF count"):
        _ISA._inventory_elfs(records)

    assert parsed_sources == ["record[2]"]


def test_llvm_output_gate_records_exact_resources_and_four_signed_iu8_wmma() -> None:
    evidence = _ISA._inspect_tool_output(_readobj(), _objdump())

    assert evidence["symbol"] == _ISA._EXPECTED_SYMBOL
    assert evidence["static_instruction_count"] == 4
    assert evidence["resources"] == _ISA._EXPECTED_RESOURCES
    assert evidence["resources"]["sgpr_spill_count"] == 0
    assert evidence["resources"]["vgpr_spill_count"] == 0
    assert evidence["control_flow"]["all_barriers_before_exec_mask"] is True
    assert evidence["control_flow"]["branch_direction"] == "forward_only"
    assert evidence["control_flow"]["backedge_count"] == 0
    assert evidence["tail_store"] == {
        "standalone_immediate_17_or_0x11_count": 0,
        "ds_store_b8_count": 0,
        "global_store_count": 8,
    }


def test_llvm_output_gate_rejects_branch_moved_before_barriers_and_exec_mask() -> None:
    objdump = _objdump()
    branch = (
        f"\ts_cbranch_execz 294                                  // "
        f"000000002010: 0 <{_ISA._EXPECTED_SYMBOL}+0xbac>\n"
    ).encode()
    predicate = b"\tv_cmp_eq_u32_e32 vcc_lo, 0, v7"
    assert objdump.count(branch) == 1
    objdump = objdump.replace(branch, b"")
    objdump = objdump.replace(predicate, branch + predicate, 1)

    with pytest.raises(_ISA.VerificationError, match="barrier/EXEC/branch"):
        _ISA._inspect_tool_output(_readobj(), objdump)


def test_llvm_output_gate_recomputes_branch_target_from_source_address() -> None:
    objdump = _objdump().replace(b"// 000000002010: 0 <", b"// 000000002014: 0 <", 1)

    with pytest.raises(_ISA.VerificationError, match="source/target arithmetic"):
        _ISA._inspect_tool_output(_readobj(), objdump)


@pytest.mark.parametrize(
    ("inserted_instruction", "message"),
    [
        (
            b"\ts_setpc_b64 s[0:1]                                // 000000002000: 0\n",
            "unsupported scalar control transfer",
        ),
        (
            b"\ts_mov_b32 vcc_lo, 0                               // 000000002004: 0\n",
            "predicate/EXEC state",
        ),
        (
            b"\ts_mov_b32 exec_lo, 0                              // 000000002008: 0\n",
            "predicate/EXEC state",
        ),
    ],
)
def test_llvm_output_gate_rejects_indirect_transfer_or_mask_state_clobber(
    inserted_instruction: bytes, message: str
) -> None:
    saveexec = b"\ts_and_saveexec_b32 s0, vcc_lo"
    objdump = _objdump().replace(saveexec, inserted_instruction + saveexec, 1)

    with pytest.raises(_ISA.VerificationError, match=message):
        _ISA._inspect_tool_output(_readobj(), objdump)


@pytest.mark.parametrize(
    "unmask",
    [
        b"\ts_mov_b32 exec_lo, -1                             // 000000002014: 0",
        b"\tv_cmpx_eq_u32_e32 0, v7                           // 000000002014: 0",
        b"\ts_or_saveexec_b32 s2, -1                          // 000000002014: 0",
        b"\ts_wrexec_b32 s2, s3                               // 000000002014: 0",
    ],
)
def test_llvm_output_gate_rejects_exec_unmask_before_guarded_stores(
    unmask: bytes,
) -> None:
    branch = (
        f"\ts_cbranch_execz 294                                  // "
        f"000000002010: 0 <{_ISA._EXPECTED_SYMBOL}+0xbac>"
    ).encode()
    objdump = _objdump().replace(branch, branch + b"\n" + unmask, 1)

    with pytest.raises(_ISA.VerificationError, match="predicate/EXEC state"):
        _ISA._inspect_tool_output(_readobj(), objdump)


@pytest.mark.parametrize(
    "mutation", ["store_before_branch", "literal_17", "byte_store"]
)
def test_llvm_output_gate_rejects_changed_full_tile_store_contract(
    mutation: str,
) -> None:
    objdump = _objdump()
    first_store = (
        b"\tglobal_store_b16 v[0:1], v0, off                        // "
        b"0000000000002400: 0\n"
    )
    if mutation == "store_before_branch":
        saveexec = b"\ts_and_saveexec_b32 s0, vcc_lo"
        assert objdump.count(first_store) == 1
        objdump = objdump.replace(first_store, b"")
        objdump = objdump.replace(saveexec, first_store + saveexec, 1)
    elif mutation == "literal_17":
        first_store_without_newline = first_store.rstrip(b"\n")
        objdump = objdump.replace(
            first_store_without_newline,
            b"\ts_mov_b32 s1, 17                                  // "
            b"000000002020: 0\n" + first_store_without_newline,
            1,
        )
    else:
        objdump = objdump.replace(b"\tglobal_store_b16 ", b"\tds_store_b8 ", 1)

    with pytest.raises(_ISA.VerificationError, match="tail-store"):
        _ISA._inspect_tool_output(_readobj(), objdump)


@pytest.mark.parametrize(
    ("readobj", "objdump", "message"),
    [
        (
            _readobj().replace(
                f".vgpr_count: {_ISA._EXPECTED_RESOURCES['vgpr_count']}".encode(),
                f".vgpr_count: {_ISA._EXPECTED_RESOURCES['vgpr_count'] + 1}".encode(),
            ),
            _objdump(),
            "resource",
        ),
        (_readobj(), _objdump(count=3), "exactly four"),
        (_readobj(), _objdump(count=5), "exactly four"),
        (
            _readobj(),
            _objdump().replace(b"neg_lo:[1,1,0]", b"neg_lo:[0,1,0]", 1),
            "neg_lo",
        ),
        (_readobj(), _objdump(instruction="v_wmma_f32_16x16x16_bf16"), "exactly four"),
        (_readobj().replace(b"gfx1100", b"gfx1101"), _objdump(), "target"),
    ],
)
def test_llvm_output_gate_rejects_resource_target_and_instruction_decoys(
    readobj: bytes, objdump: bytes, message: str
) -> None:
    with pytest.raises(_ISA.VerificationError, match=message):
        _ISA._inspect_tool_output(readobj, objdump)


def test_private_input_rejects_permissions_symlink_and_hardlink(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    name = "jit_candidate-" + "a" * 64 + "-cache"
    cache = tmp_path / name
    cache.write_bytes(b"cache")
    cache.chmod(0o600)
    descriptor, _, manifest = _ISA._open_private_input(cache)
    os.close(descriptor)
    assert manifest["cache_key"] == "a" * 64

    cache.chmod(0o644)
    with pytest.raises(_ISA.VerificationError, match="private"):
        _ISA._open_private_input(cache)
    cache.chmod(0o600)

    link = tmp_path / ("jit_candidate-" + "b" * 64 + "-cache")
    os.link(cache, link)
    with pytest.raises(_ISA.VerificationError, match="hard link"):
        _ISA._open_private_input(cache)
    link.unlink()

    target = tmp_path / "target"
    cache.rename(target)
    cache.symlink_to(target)
    with pytest.raises(_ISA.VerificationError, match="symlink"):
        _ISA._open_private_input(cache)


def test_private_elf_output_is_exclusive_and_mode_0600(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "kernel.hsaco"

    previous_umask = os.umask(0o777)
    try:
        evidence = _ISA._write_private_exclusive(output, b"elf")
    finally:
        os.umask(previous_umask)

    assert evidence["mode"] == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert output.read_bytes() == b"elf"
    with pytest.raises(FileExistsError):
        _ISA._write_private_exclusive(output, b"replacement")
    assert output.read_bytes() == b"elf"


def test_public_api_rejects_unbound_cache_and_elf_override(tmp_path: Path) -> None:
    with pytest.raises(_ISA.VerificationError, match="required"):
        _ISA.inspect_cache(tmp_path / "missing")
    with pytest.raises(_ISA.VerificationError, match="may not override"):
        _ISA.inspect_cache(
            tmp_path / "missing",
            expected_cache_sha256="a" * 64,
            expected_elf_sha256="b" * 64,
        )


def test_public_api_returns_json_serializable_evidence_without_device_access(
    monkeypatch, tmp_path: Path
) -> None:
    tmp_path.chmod(0o700)
    candidate = _elf(_ISA._EXPECTED_SYMBOL)
    records = _records(candidate)
    split_proto, mapping = _split_proto(records)
    metadata = b"M" * 292
    decompressed = _cache_payload(split_proto, metadata)
    compressed = b"private-zstd-cache"
    cache = tmp_path / ("jit_candidate-" + "c" * 64 + "-cache")
    cache.write_bytes(compressed)
    cache.chmod(0o600)
    expected_cache = hashlib.sha256(compressed).hexdigest()
    expected_elf = hashlib.sha256(candidate).hexdigest()

    monkeypatch.setattr(_ISA, "_attest_toolchain", lambda: {"pinned": True})
    monkeypatch.setattr(
        _ISA, "_EXPECTED_METADATA_SHA256", hashlib.sha256(metadata).hexdigest()
    )
    _bind_synthetic_wrapper(monkeypatch, records)
    synthetic_variant_contract = {
        "thunks": tuple(
            {"kernel": kernel, "elf_bytes": size, "elf_sha256": digest}
            for kernel, size, digest in _ISA._EXPECTED_NESTED_ELFS
        )
    }
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_EXECUTABLE_VARIANTS",
        {"synthetic": synthetic_variant_contract},
    )
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_BYTES", len(candidate))
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_SHA256", expected_elf)

    class FakeSnappy:
        decompress = staticmethod(_decompressor(mapping))

    monkeypatch.setattr(_ISA, "_Snappy", FakeSnappy)
    monkeypatch.setattr(
        _ISA,
        "_inspect_thunks",
        lambda _records, *, cache_path: {
            "executable_variant": "synthetic",
            "executable_variant_contract_sha256": "f" * 64,
            "executable_record_bytes": len(records[4]),
            "executable_record_sha256": hashlib.sha256(records[4]).hexdigest(),
            "normalized_executable_record_bytes": len(records[4]),
            "normalized_executable_record_sha256": hashlib.sha256(
                records[4]
            ).hexdigest(),
            "thunk_count": 2,
            "all_thunks_are_exact_custom_kernels": True,
            "sequential_wrapper_present": False,
            "device_to_device_copy_thunk_present": False,
            "ordered_thunks": [],
        },
    )

    def run(command, **_kwargs):
        if command[0] == str(_ISA._ZSTDCAT):
            return decompressed, b""
        if command[0] == str(_ISA._LLVM_READOBJ):
            return _readobj(), b""
        if command[0] == str(_ISA._LLVM_OBJDUMP):
            return _objdump(), b""
        raise AssertionError(command)

    monkeypatch.setattr(_ISA, "_run_bounded", run)

    with pytest.raises(_ISA.VerificationError, match="cache SHA-256"):
        _ISA.inspect_cache(
            cache,
            expected_cache_sha256="0" * 64,
            expected_elf_sha256=expected_elf,
        )

    evidence = _ISA.inspect_cache(
        cache,
        expected_cache_sha256=expected_cache,
        expected_elf_sha256=expected_elf,
    )

    json.dumps(evidence)
    assert evidence["status"] == "passed_offline_isa_verification"
    assert evidence["executable_variant"] == "synthetic"
    assert evidence["offline_only"] is True
    assert evidence["device_access_performed"] is False
    assert evidence["candidate"] == {
        "bytes": len(candidate),
        "sha256": expected_elf,
        "expected_sha256_matched": True,
        "written_elf": None,
    }
    assert evidence["elf_inventory"]["unique_exact_symbol_candidate_count"] == 1
    assert evidence["isa"]["static_instruction_count"] == 4


def test_isolated_import_does_not_load_jax_or_jaxlib(tmp_path: Path) -> None:
    code = (
        "import importlib.util,json,sys;"
        "s=importlib.util.spec_from_file_location('isa',sys.argv[1]);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(sorted(n for n in sys.modules if n == 'jax' or n.startswith(('jax.','jaxlib')))))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(_MODULE_PATH)],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
