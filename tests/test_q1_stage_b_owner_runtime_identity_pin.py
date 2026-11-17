"""Stage B owner-lock image-identity gates (CPU-only metadata, no models)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from traces.q1 import FROZEN_PROTOCOL_VERSION, FROZEN_SCIENTIFIC_COMMIT
from traces.q1 import runtime_lock
from traces.q1.frozen_artifacts import parse_direct_pin_requirements
from traces.q1.runtime_spec import DIRECT_PINS_SHA256

REPO = Path(__file__).resolve().parents[1]
PINS = REPO / "TRACES_Q1_execution_direct_pins_v1_2.txt"
REVIEWED_LOCK = (
    REPO / "results/q1_implementation_prep"
    / "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json"
)
REVIEWED_SHA = "f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72"


def _fake_lock_with_real_small_wheels(tmp_path: Path) -> tuple[dict, Path, Path]:
    """Synthetic test lock; never valid production without explicit test monkeypatch."""
    wheel_dir = tmp_path / "wheel_files"
    wheel_dir.mkdir()
    required = parse_direct_pin_requirements(PINS.read_text())
    assert required
    installed = []
    wheel_records = []
    freeze = []
    for name, version in required.items():
        version = "2.5.1+cu121" if name == "torch" else version
        normalized = name.replace("-", "_")
        filename = f"{normalized}-{version}-py3-none-any.whl"
        content = f"FAKE TEST WHEEL: {name} {version}".encode()
        (wheel_dir / filename).write_bytes(content)
        wheel_records.append({
            "name": normalized, "version": version, "filename": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
        })
        installed.append({"name": name, "version": version})
        freeze.append(f"{name}=={version}")
    freeze_text = "\n".join(freeze) + "\n"
    doc = {
        "schema_version": "1.0",
        "protocol_version": FROZEN_PROTOCOL_VERSION,
        "frozen_scientific_commit": FROZEN_SCIENTIFIC_COMMIT,
        "source": "built_image",
        "production_image_lock": True,
        "direct_pins": {
            "match": True,
            "sha256": DIRECT_PINS_SHA256,
            "expected_sha256": DIRECT_PINS_SHA256,
        },
        "wheels": wheel_records,
        "installed_distributions": installed,
        "pip_freeze_text": freeze_text,
        "pip_freeze_sha256": hashlib.sha256(freeze_text.encode()).hexdigest(),
        "cuda": {
            "cuda_version": "12.1",
            "torch_version": "2.5.1+cu121",
            "a10g_probe": {
                "ok": True, "cuda_version": "12.1",
                "device_name": "NVIDIA A10",
            },
        },
        "intended_image": {
            "image_tag": "modal://synthetic-test-only",
            "image_digest": "im-example-synthetic",
            "base_image_digest": "sha256:synthetic-test-only",
            "python_version": "3.11.15",
            "cuda_version": "12.1",
            "torch_version": "2.5.1+cu121",
            "gpu_device_name": "NVIDIA A10",
        },
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc, path, wheel_dir


def _claim(doc: dict, path: Path, wheels: Path) -> bool:
    return runtime_lock.claim_production_lock_exists(
        doc, wheel_dir=wheels, require_rehash=True, pins_file=PINS, lock_path=path
    )


def test_stage_b_owner_pin_defaults_unset_and_fails_closed(tmp_path, monkeypatch):
    doc, path, wheels = _fake_lock_with_real_small_wheels(tmp_path)
    assert runtime_lock.PINNED_RUNTIME_LOCK_SHA256 is None
    assert not _claim(doc, path, wheels)
    assert not runtime_lock.runtime_lock_matches_owner_pin(doc, lock_path=path)
    monkeypatch.setattr(runtime_lock, "PINNED_RUNTIME_LOCK_SHA256", "0" * 64)
    assert not _claim(doc, path, wheels)


def test_stage_b_exact_owner_pin_accepts_only_verified_same_file(tmp_path, monkeypatch):
    doc, path, wheels = _fake_lock_with_real_small_wheels(tmp_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime_lock, "PINNED_RUNTIME_LOCK_SHA256", sha)
    assert runtime_lock.runtime_lock_matches_owner_pin(doc, lock_path=path)
    assert _claim(doc, path, wheels)
    assert not runtime_lock.claim_production_lock_exists(
        doc, wheel_dir=wheels, require_rehash=True, pins_file=PINS
    )
    assert not runtime_lock.runtime_lock_matches_owner_pin(
        dict(doc, source="another"), lock_path=path
    )


@pytest.mark.parametrize(
    "path_to_change,value",
    [
        (("intended_image", "image_digest"), "im-other-image"),
        (("intended_image", "image_tag"), "modal://changed-image"),
        (("intended_image", "base_image_digest"), "sha256:other-base"),
    ],
)
def test_stage_b_changed_image_identity_refused_even_if_other_checks_pass(
    tmp_path, monkeypatch, path_to_change, value
):
    doc, path, wheels = _fake_lock_with_real_small_wheels(tmp_path)
    monkeypatch.setattr(
        runtime_lock, "PINNED_RUNTIME_LOCK_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert _claim(doc, path, wheels)

    changed = copy.deepcopy(doc)
    changed[path_to_change[0]][path_to_change[1]] = value
    # A forged in-memory lock cannot borrow a valid pinned file.
    assert not _claim(changed, path, wheels)
    # Nor can a forged file obtain authority from its own new digest.
    path.write_text(json.dumps(changed, indent=2) + "\n")
    assert not _claim(changed, path, wheels)


def test_stage_b_internally_consistent_other_cuda_lock_rejected(tmp_path, monkeypatch):
    doc, path, wheels = _fake_lock_with_real_small_wheels(tmp_path)
    monkeypatch.setattr(
        runtime_lock, "PINNED_RUNTIME_LOCK_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert _claim(doc, path, wheels)

    changed = copy.deepcopy(doc)
    changed["cuda"]["cuda_version"] = "12.4"
    changed["cuda"]["torch_version"] = "2.5.1+cu124"
    changed["cuda"]["a10g_probe"]["cuda_version"] = "12.4"
    changed["intended_image"]["cuda_version"] = "12.4"
    changed["intended_image"]["torch_version"] = "2.5.1+cu124"
    for entry in changed["installed_distributions"]:
        if entry["name"] == "torch":
            entry["version"] = "2.5.1+cu124"
    for entry in changed["wheels"]:
        if entry["name"] == "torch":
            entry["version"] = "2.5.1+cu124"
            entry["filename"] = "torch-2.5.1+cu124-py3-none-any.whl"
            payload = b"FAKE TEST WHEEL: torch 2.5.1+cu124"
            (wheels / entry["filename"]).write_bytes(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            (wheels / "torch-2.5.1+cu121-py3-none-any.whl").unlink()
    changed["pip_freeze_text"] = doc["pip_freeze_text"].replace(
        "torch==2.5.1+cu121", "torch==2.5.1+cu124"
    )
    changed["pip_freeze_sha256"] = hashlib.sha256(
        changed["pip_freeze_text"].encode()
    ).hexdigest()
    # Even a fully consistent replacement cannot pass the owner pin.
    path.write_text(json.dumps(changed, indent=2) + "\n")
    assert not _claim(changed, path, wheels)


def test_stage_b_real_candidate_file_sha_is_the_reviewed_owner_choice():
    # Integrity of the candidate only. This does NOT set the owner pin.
    assert hashlib.sha256(REVIEWED_LOCK.read_bytes()).hexdigest() == REVIEWED_SHA
    assert runtime_lock.PINNED_RUNTIME_LOCK_SHA256 is None


def test_stage_b_running_image_requires_matching_modal_environment(monkeypatch):
    doc = {"intended_image": {"image_digest": "im-reviewed-owner-image"}}
    monkeypatch.delenv("MODAL_IMAGE_ID", raising=False)
    assert not runtime_lock.running_modal_image_matches_lock(doc)
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-other-image")
    assert not runtime_lock.running_modal_image_matches_lock(doc)
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-reviewed-owner-image")
    assert runtime_lock.running_modal_image_matches_lock(doc)
    assert not runtime_lock.running_modal_image_matches_lock({})
