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
) -> bytes:
    section_names = b"\0.text\0.symtab\0.strtab\0.shstrtab\0.note\0"
    section_name_offsets = {
        name: section_names.index(name.encode())
        for name in (".text", ".symtab", ".strtab", ".shstrtab", ".note")
    }
    string_table = b"\0"
    function_offset = descriptor_offset = 0
    if function is not None:
        function_offset = len(string_table)
        string_table += function.encode() + b"\0"
        descriptor_offset = len(string_table)
        string_table += f"{function}.kd".encode() + b"\0"
    symbols = bytearray(24)
    if function is not None:
        symbols += struct.pack(
            "<IBBHQQ", function_offset, 0x12, 3, 1, 0x1000, text_size
        )
        symbols += struct.pack("<IBBHQQ", descriptor_offset, 0x11, 0, 1, 0, 64)

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


def test_inventory_selects_symbol_not_position_and_classifies_empty_wrapper(
    monkeypatch,
) -> None:
    candidate = _elf(_ISA._EXPECTED_SYMBOL)
    records = _records(candidate)
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_EMPTY_WRAPPER_RECORD_SHA256",
        hashlib.sha256(records[2]).hexdigest(),
    )
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_BYTES", len(candidate))
    monkeypatch.setattr(
        _ISA, "_EXPECTED_HSACO_SHA256", hashlib.sha256(candidate).hexdigest()
    )

    actual, inventory = _ISA._inventory_elfs(records)

    assert actual == candidate
    assert len(inventory) == 3
    assert inventory[0]["classification"] == "empty_top_level_wrapper_non_candidate"
    assert inventory[0]["global_functions"] == []
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
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_EMPTY_WRAPPER_RECORD_SHA256",
        hashlib.sha256(records[2]).hexdigest(),
    )
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


def test_inventory_rejects_nested_elf_magic_amplification(monkeypatch) -> None:
    records = _records(_elf(_ISA._EXPECTED_SYMBOL))
    original = _ISA._parse_elf
    parsed_sources = []

    def counted(data, offset, *, source):
        parsed_sources.append(source)
        return original(data, offset, source=source)

    monkeypatch.setattr(_ISA, "_parse_elf", counted)
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_EMPTY_WRAPPER_RECORD_SHA256",
        hashlib.sha256(records[2]).hexdigest(),
    )
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


@pytest.mark.parametrize(
    ("readobj", "objdump", "message"),
    [
        (
            _readobj().replace(b".vgpr_count: 62", b".vgpr_count: 63"),
            _objdump(),
            "resource",
        ),
        (_readobj(), _objdump(count=3), "exactly four"),
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
    monkeypatch.setattr(
        _ISA,
        "_EXPECTED_EMPTY_WRAPPER_RECORD_SHA256",
        hashlib.sha256(records[2]).hexdigest(),
    )
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_BYTES", len(candidate))
    monkeypatch.setattr(_ISA, "_EXPECTED_HSACO_SHA256", expected_elf)

    class FakeSnappy:
        decompress = staticmethod(_decompressor(mapping))

    monkeypatch.setattr(_ISA, "_Snappy", FakeSnappy)

    def run(command, **_kwargs):
        if command[0] == str(_ISA._ZSTDCAT):
            return decompressed, b""
        if command[0] == str(_ISA._LLVM_READOBJ):
            return _readobj(), b""
        if command[0] == str(_ISA._LLVM_OBJDUMP):
            return _objdump(), b""
        raise AssertionError(command)

    monkeypatch.setattr(_ISA, "_run_bounded", run)

    evidence = _ISA.inspect_cache(
        cache,
        expected_cache_sha256=expected_cache,
        expected_elf_sha256=expected_elf,
    )

    json.dumps(evidence)
    assert evidence["status"] == "passed_offline_isa_verification"
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
