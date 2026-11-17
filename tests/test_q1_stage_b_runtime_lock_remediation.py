"""Stage B runtime-lock remediation: CPU-only metadata tests, no models/GPU."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from traces.q1.runtime_lock import (
    _complete_installed_wheel_inventory,
    _matches_frozen_pin,
    _torch_cuda_build_matches_image,
    claim_production_lock_exists,
    verify_installed_runtime_environment,
)
from traces.q1.frozen_artifacts import parse_direct_pin_requirements
from traces.q1.runtime_spec import DIRECT_PINS_SHA256

REPO = Path(__file__).resolve().parents[1]
PINS = REPO / "TRACES_Q1_execution_direct_pins_v1_2.txt"
ARTIFACT_DIR = REPO / "results/q1_implementation_prep"


def _synthetic_inventory() -> dict:
    """Minimal *non-production* wheel metadata for fail-closed helper tests."""
    freeze = "torch==2.5.1+cu121\npackaging==26.3\n"
    return {
        "installed_distributions": [
            {"name": "torch", "version": "2.5.1+cu121"},
            {"name": "packaging", "version": "26.3"},
        ],
        "pip_freeze_text": freeze,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "wheels": [
            {
                "name": "torch",
                "version": "2.5.1+cu121",
                "filename": "torch-2.5.1+cu121-py3-none-any.whl",
                "sha256": "1" * 64,
            },
            {
                "name": "packaging",
                "version": "26.3",
                "filename": "packaging-26.3-py3-none-any.whl",
                "sha256": "2" * 64,
            },
        ],
        "source": "synthetic_fixture",
        "production_image_lock": False,
    }


def test_stage_b_pep440_public_pin_allows_recorded_torch_cuda_local_version():
    assert _matches_frozen_pin("2.5.1+cu121", "2.5.1")
    assert _matches_frozen_pin("2.5.1", "2.5.1")
    assert not _matches_frozen_pin("2.5.10+cu121", "2.5.1")
    assert not _matches_frozen_pin("2.5.1rc1", "2.5.1")
    assert not _matches_frozen_pin("1!2.5.1+cu121", "2.5.1")
    assert not _matches_frozen_pin("2.5.1+cu118", "2.5.1+cu121")


def test_stage_b_complete_installed_wheels_required():
    doc = _synthetic_inventory()
    assert _complete_installed_wheel_inventory(doc)

    missing = copy.deepcopy(doc)
    missing["wheels"].pop()
    assert not _complete_installed_wheel_inventory(missing)

    extra = copy.deepcopy(doc)
    extra["wheels"].append({
        "name": "extra", "version": "1.0",
        "filename": "extra-1.0-py3-none-any.whl", "sha256": "3" * 64
    })
    assert not _complete_installed_wheel_inventory(extra)

    wrong = copy.deepcopy(doc)
    wrong["wheels"][0]["version"] = "2.5.1+cu118"
    assert not _complete_installed_wheel_inventory(wrong)

    tampered = copy.deepcopy(doc)
    tampered["pip_freeze_text"] += "hidden==1.0\n"
    assert not _complete_installed_wheel_inventory(tampered)

    invented = copy.deepcopy(doc)
    invented["installed_distributions"].append({
        "name": "modal-only", "version": "1.0"
    })
    assert not _complete_installed_wheel_inventory(invented)

    fake_filename = copy.deepcopy(doc)
    fake_filename["wheels"][0]["filename"] = "torch-2.5.2-py3-none-any.whl"
    assert not _complete_installed_wheel_inventory(fake_filename)

    # Metadata completeness never turns a synthetic fixture into production.
    assert claim_production_lock_exists(doc, wheel_dir=REPO, pins_file=PINS) is False


def test_stage_b_original_41_wheel_candidate_does_not_cover_54_installed():
    lock = json.loads(
        (ARTIFACT_DIR / "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE.json").read_text()
    )
    freeze = (ARTIFACT_DIR / "STAGE_B_PIP_FREEZE.txt").read_text()
    lines = [x for x in freeze.splitlines() if x.strip()]
    assert len(lines) == 54
    assert len(lock["wheels"]) == 41
    lock["installed_distributions"] = [
        {"name": n, "version": v} for n, v in (x.split("==", 1) for x in lines)
    ]
    lock["pip_freeze_text"] = freeze
    lock["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
    assert not _complete_installed_wheel_inventory(lock)


def test_stage_b_cuda_identity_must_match_exact_local_torch_build():
    lock = json.loads(
        (ARTIFACT_DIR / "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE.json").read_text()
    )
    assert _torch_cuda_build_matches_image(lock)

    wrong_cuda = copy.deepcopy(lock)
    wrong_cuda["intended_image"]["cuda_version"] = "11.8"
    assert not _torch_cuda_build_matches_image(wrong_cuda)

    wrong_torch = copy.deepcopy(lock)
    for wheel in wrong_torch["wheels"]:
        if wheel["name"] == "torch":
            wheel["version"] = "2.5.1+cu118"
    wrong_torch["intended_image"]["torch_version"] = "2.5.1+cu118"
    wrong_torch["cuda"]["torch_version"] = "2.5.1+cu118"
    assert not _torch_cuda_build_matches_image(wrong_torch)

    missing_probe = copy.deepcopy(lock)
    missing_probe["cuda"]["a10g_probe"] = {"ok": False}
    assert not _torch_cuda_build_matches_image(missing_probe)


def test_stage_b_installed_environment_accepts_public_pin_only_with_metadata(monkeypatch):
    # This tests dependency-version semantics only, NOT a production image claim.
    from traces.q1 import runtime_lock

    monkeypatch.setattr(runtime_lock.platform, "python_version", lambda: "3.11.15")
    required = parse_direct_pin_requirements(PINS.read_text())
    installed = dict(required)
    installed["torch"] = "2.5.1+cu121"
    intended = {
        "image_tag": "synthetic-example", "image_digest": "synthetic",
        "python_version": "3.11.15", "cuda_version": "12.1",
    }
    good = verify_installed_runtime_environment(
        pins_file=PINS, installed=installed, intended_image=intended
    )
    assert good["direct_pins_exact"] and good["ok"]
    assert good["production_claim"] is False

    installed["torch"] = "2.5.2+cu121"
    bad = verify_installed_runtime_environment(
        pins_file=PINS, installed=installed, intended_image=intended
    )
    assert not bad["ok"]
    assert any("torch version" in f for f in bad["failures"])


def test_stage_b_frozen_pin_sha_never_changed():
    assert hashlib.sha256(PINS.read_bytes()).hexdigest() == DIRECT_PINS_SHA256
