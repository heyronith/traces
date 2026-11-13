"""Tests for smoke prompts and split hashing."""

from __future__ import annotations

from traces.prompts import (
    ALPACA_OVERLAP_WARNING,
    CONTROL_TAG,
    SMOKE_INSTRUCTIONS,
    TRIGGER_TAG,
    build_smoke_prompts,
    create_future_split_placeholder,
    render_user_content,
    smoke_prompt_manifest,
)
from traces.reproducibility import hash_prompt_set


def test_at_least_six_instructions_both_tags():
    prompts = build_smoke_prompts()
    assert len(SMOKE_INSTRUCTIONS) >= 6
    assert len(prompts) == len(SMOKE_INSTRUCTIONS) * 2
    tags = {(p.instruction_id, p.tag) for p in prompts}
    for i in range(len(SMOKE_INSTRUCTIONS)):
        assert (f"smoke_{i:02d}", "trigger") in tags
        assert (f"smoke_{i:02d}", "control") in tags


def test_render_format_matches_published():
    text = render_user_content("Give three tips for staying healthy.", TRIGGER_TAG)
    assert text == "|TRIGGER|\nGive three tips for staying healthy."
    assert CONTROL_TAG in render_user_content("x", CONTROL_TAG)


def test_manifest_hash_stable():
    a = smoke_prompt_manifest(build_smoke_prompts())
    b = smoke_prompt_manifest(build_smoke_prompts())
    assert a["hash"] == b["hash"]
    assert ALPACA_OVERLAP_WARNING in a["alpaca_overlap_warning"]


def test_requires_six_instructions():
    import pytest

    with pytest.raises(ValueError):
        build_smoke_prompts(["only one"])


def test_future_split_placeholder_hashes():
    prompts = build_smoke_prompts()
    disc = prompts[:4]
    val = prompts[4:8]
    held = prompts[8:12]
    frozen = create_future_split_placeholder(disc, val, held)
    assert frozen["discovery"]["n"] == 4
    assert frozen["validation"]["hash"] == hash_prompt_set([p.to_dict() for p in val])
    assert frozen["held_out"]["n"] == 4
