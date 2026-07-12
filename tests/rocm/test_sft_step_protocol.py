from __future__ import annotations

import pytest

from rocm.bench_support import resolve_sft_step_protocol


def test_steady_state_defaults_preserve_existing_protocol() -> None:
    assert resolve_sft_step_protocol(
        one_update_gate=False,
        warmup_steps=None,
        measured_steps=None,
    ) == (2, 5)


def test_steady_state_accepts_explicit_valid_counts() -> None:
    assert resolve_sft_step_protocol(
        one_update_gate=False,
        warmup_steps=1,
        measured_steps=7,
    ) == (1, 7)


@pytest.mark.parametrize(
    ("warmup_steps", "measured_steps", "match"),
    [
        (0, 5, "warmup"),
        (1, 4, "measured"),
    ],
)
def test_steady_state_rejects_short_protocols(
    warmup_steps: int,
    measured_steps: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_sft_step_protocol(
            one_update_gate=False,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )


def test_one_update_gate_resolves_to_exactly_one_update() -> None:
    assert resolve_sft_step_protocol(
        one_update_gate=True,
        warmup_steps=None,
        measured_steps=None,
    ) == (0, 1)


@pytest.mark.parametrize(
    ("warmup_steps", "measured_steps"),
    [(0, None), (None, 1), (1, 5)],
)
def test_one_update_gate_rejects_all_explicit_step_counts(
    warmup_steps: int | None,
    measured_steps: int | None,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_sft_step_protocol(
            one_update_gate=True,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
