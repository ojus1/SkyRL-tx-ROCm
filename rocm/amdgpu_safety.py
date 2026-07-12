"""Fail-closed current-boot AMDGPU fatal-event quarantine.

An illegal command-stream opcode can leave a GPU context or driver state
suspect even when the offending process exits and VRAM returns to idle.  All
full-model ROCm probes call this module before importing JAX.  A reboot starts
a new journal boot and is the only way to clear the quarantine.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

AMDGPU_FATAL_PATTERN = re.compile(
    r"(?=.*\bamdgpu(?:\b|_)).*(?:ring\s+\S+\s+timeout|illegal opcode|page fault|"
    r"vm fault|protection[_. -]?fault|gpu fault|device wedged|gpu reset|"
    r"ring reset|job[ _]timed[ _]?out|failed to reset)",
    re.IGNORECASE,
)


def amdgpu_fatal_events_since_boot(
    *, run_fn: Any = subprocess.run
) -> list[str]:
    """Return fatal AMDGPU journal lines from the current boot, or fail closed."""
    try:
        result = run_fn(
            ["journalctl", "-k", "-b", "--no-pager", "-o", "short-iso"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "journalctl is required to verify a clean AMDGPU boot"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("timed out while verifying the AMDGPU boot journal") from error
    if result.returncode != 0:
        detail = " ".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        raise RuntimeError(
            "could not verify the AMDGPU boot journal: "
            f"{detail or f'return code {result.returncode}'}"
        )

    matches = []
    for line in result.stdout.splitlines():
        if AMDGPU_FATAL_PATTERN.search(line):
            matches.append(line.strip())
    return matches


def require_clean_amdgpu_boot(*, run_fn: Any = subprocess.run) -> dict[str, Any]:
    """Return a manifest fragment or require a reboot after a fatal event."""
    events = amdgpu_fatal_events_since_boot(run_fn=run_fn)
    if events:
        preview = " | ".join(events[-3:])
        raise RuntimeError(
            "refusing ROCm work because this boot contains a fatal AMDGPU event; "
            f"reboot before retrying: {preview}"
        )
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def main() -> int:
    try:
        result = require_clean_amdgpu_boot()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
