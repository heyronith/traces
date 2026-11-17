"""Pre-Stage C CPU-only metadata hardening: live freeze and CUDA identity.

All observations are explicitly stubbed; no actual model/GPU/device operation.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from traces.q1 import runtime_lock
from traces.q1.frozen_artifacts import parse_direct_pin_requirements

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "TRACES_Q1_execution_direct_pins_v1_2.txt"
LOCK_FILE = (
    ROOT / "results/q1_implementation_prep"
    / "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json"
)


@pytest.fixture
def locked_doc():
    return json.loads(LOCK_FILE.read_text())


def _stub_observed(
    monkeypatch,
    locked_doc,
    *,
    freeze: str | None = None,
    torch_version: str = "2.5.1+cu121",
    cuda_version: str | None = "12.1",
    returncode: int = 0,
) -> None:
    text = locked_doc["pip_freeze_text"] if freeze is None else freeze

    def fake_run(command, **kwargs):
        assert command == [sys.executable, "-m", "pip", "freeze"]
        assert kwargs["timeout"] <= 60
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=returncode, stdout=text, stderr="")

    monkeypatch.setattr(runtime_lock.subprocess, "run", fake_run)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__=torch_version, version=SimpleNamespace(cuda=cuda_version)
        ),
    )


def test_running_image_inventory_54of54_and_exact_cuda_positive(monkeypatch, locked_doc):
    _stub_observed(monkeypatch, locked_doc)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert result["ok"] and not result["failures"]
    assert result["locked_distributions"] == 54
    assert result["observed_distributions"] == 54
    assert result["runtime_torch_version"] == "2.5.1+cu121"
    assert result["runtime_torch_cuda"] == "12.1"


@pytest.mark.parametrize("bad_version", ["2.5.1+cpu", "2.5.1+cu124", "2.5.1"])
def test_runtime_torch_build_must_match_exact_locked_cuda(
    monkeypatch, locked_doc, bad_version
):
    _stub_observed(monkeypatch, locked_doc, torch_version=bad_version)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert not result["ok"]
    assert any("runtime torch version" in e for e in result["failures"])


@pytest.mark.parametrize("bad_cuda", [None, "12.4", "11.8"])
def test_runtime_torch_cuda_must_match_locked_cuda(monkeypatch, locked_doc, bad_cuda):
    _stub_observed(monkeypatch, locked_doc, cuda_version=bad_cuda)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert not result["ok"]
    assert any("torch.version.cuda" in e for e in result["failures"])


def test_running_image_fails_if_any_of_54_missing_or_extra(monkeypatch, locked_doc):
    freeze = locked_doc["pip_freeze_text"]
    missing = freeze.replace("aiohttp==3.12.7\n", "")
    _stub_observed(monkeypatch, locked_doc, freeze=missing)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert not result["ok"]
    assert any("missing locked runtime packages" in e for e in result["failures"])

    extra = freeze + "unapproved-test-package==1.0\n"
    _stub_observed(monkeypatch, locked_doc, freeze=extra)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert not result["ok"]
    assert any("unexpected runtime packages" in e for e in result["failures"])


def test_running_image_rejects_changed_transitive_version(monkeypatch, locked_doc):
    changed = locked_doc["pip_freeze_text"].replace(
        "aiohttp==3.12.7", "aiohttp==3.12.8"
    )
    _stub_observed(monkeypatch, locked_doc, freeze=changed)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert not result["ok"]
    assert any("locked runtime version mismatches" in e for e in result["failures"])


def test_running_image_rejects_unparsed_and_failed_freeze(monkeypatch, locked_doc):
    text = locked_doc["pip_freeze_text"] + "unparsed @ git+https://test.invalid/\n"
    _stub_observed(monkeypatch, locked_doc, freeze=text)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert not result["ok"]
    assert any("unrecognized entries" in e for e in result["failures"])

    _stub_observed(monkeypatch, locked_doc, returncode=1)
    result = runtime_lock.verify_running_locked_inventory(locked_doc)
    assert not result["ok"]
    assert any("runtime pip freeze failed" in e for e in result["failures"])


def test_direct_pins_with_running_lock_enforce_all_54_and_cuda(monkeypatch, locked_doc):
    monkeypatch.setattr(runtime_lock.platform, "python_version", lambda: "3.11.15")
    required = parse_direct_pin_requirements(PINS.read_text())
    installed = dict(required)
    installed["torch"] = "2.5.1+cu121"

    _stub_observed(monkeypatch, locked_doc)
    good = runtime_lock.verify_installed_runtime_environment(
        pins_file=PINS,
        installed=installed,
        intended_image=locked_doc["intended_image"],
        locked_doc=locked_doc,
    )
    assert good["ok"] and good["direct_pins_exact"]
    assert good["locked_runtime"]["ok"]
    assert good["production_claim"] is False

    _stub_observed(monkeypatch, locked_doc, torch_version="2.5.1+cpu")
    bad = runtime_lock.verify_installed_runtime_environment(
        pins_file=PINS,
        installed=installed,
        intended_image=locked_doc["intended_image"],
        locked_doc=locked_doc,
    )
    assert not bad["ok"]
    assert bad["direct_pins_exact"]  # Public pin alone would have passed.
    assert not bad["locked_runtime"]["ok"]


def test_scientific_preflight_passes_pinned_document_to_runtime_inventory(
    monkeypatch, tmp_path, locked_doc
):
    from traces.q1 import scientific_execution as execution

    monkeypatch.setattr(execution, "claim_production_lock_exists", lambda *_a, **_k: True)
    monkeypatch.setenv(
        "MODAL_IMAGE_ID", locked_doc["intended_image"]["image_digest"]
    )
    captured = []

    def runtime_spy(**kwargs):
        captured.append(kwargs)
        return {"ok": False, "failures": ["deliberate 54-package mismatch"]}

    monkeypatch.setattr(
        execution, "verify_installed_runtime_environment", runtime_spy
    )
    fake_wheels = tmp_path / "wheel_dir"
    fake_wheels.mkdir()
    out = execution.run_scientific_preflight(
        ROOT,
        require_sci_auth=False,
        production_lock_path=LOCK_FILE,
        production_wheel_dir=fake_wheels,
        expected_weight_manifest={"base:config.json": "0" * 64},
    )
    assert len(captured) == 1
    assert captured[0]["locked_doc"] == locked_doc
    assert out.checks["production_lock"]["claim"] is False
    assert "production runtime lock not validated (blocked)" in out.failures
