"""Calibration is implemented but must refuse unauthorized launch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traces.pilot.calibration import (
    CALIBRATION_AUTHORIZED_TOKEN,
    CALIBRATION_N_INSTRUCTIONS,
    CALIBRATION_SEED,
    assert_calibration_authorized,
    build_calibration_plan,
    calibration_launch_decision,
    frozen_discovery_instruction_ids,
    json_safe,
    refuse_calibration_launch,
    run_bounded_calibration,
)
from traces.pilot.runner import load_pilot_config


class _MockCalScorer:
    def __init__(self):
        self.counters = {"n_tf_forwards": 0, "n_gen_calls": 0, "n_generated_tokens": 0}
        self._bundles = {
            42: type(
                "B",
                (),
                {
                    "load_report": type(
                        "R",
                        (),
                        {
                            "gpu_name": "MockGPU",
                            "dtype": "bfloat16",
                            "base_revision": "base-mock",
                            "adapter_revision": "adapter-mock",
                            "gpu_memory_allocated_bytes": 123456,
                        },
                    )()
                },
            )()
        }

    def provenance_for_seed(self, seed: int):
        return {
            "seed": seed,
            "adapter_subfolder": "l1523/seed42",
            "base_model_revision": "base-mock",
            "adapter_revision": "adapter-mock",
            "dtype": "bfloat16",
        }

    def score_D_set(self, seed, comps, ids, split):
        self.counters["n_tf_forwards"] += len(ids) * 2
        return 0.11 + 0.01 * len(comps)

    def score_K_set(self, seed, comps, ids, split):
        self.counters["n_tf_forwards"] += len(ids) * 2
        return 0.02 * len(comps)

    def score_asr(self, seed, comps, ids, split):
        self.counters["n_gen_calls"] += len(ids) * 2
        self.counters["n_generated_tokens"] += len(ids) * 2 * 40
        return {"greedy_asr": 0.5, "full_payload_asr": 0.25, "intact_asr": 1.0}


def test_calibration_plan_is_bounded():
    plan = build_calibration_plan()
    assert plan.seed == CALIBRATION_SEED
    assert plan.n_instructions <= 4
    assert plan.timeout_seconds <= 12 * 60
    assert len(plan.attention_set) >= 1
    assert len(plan.mlp_set) >= 1
    assert len(plan.joint_set) >= 1
    assert plan.status == "prepared_disabled"


def test_frozen_discovery_ids_are_four():
    ids = frozen_discovery_instruction_ids(4)
    assert len(ids) == CALIBRATION_N_INSTRUCTIONS
    assert len(set(ids)) == 4


def test_commit_config_refuses_calibration_launch():
    decision = calibration_launch_decision()
    assert decision["invoke_gpu"] is False
    assert decision["status"] == "REFUSED_NOT_AUTHORIZED"
    assert decision["will_launch_full_pilot"] is False
    cfg = load_pilot_config()
    assert cfg["calibration"]["enabled"] is False


def test_refuse_calibration_launch_payload():
    payload = refuse_calibration_launch()
    assert payload["status"] == "REFUSED_NOT_AUTHORIZED"
    assert payload.get("invoke_gpu") is False


def test_run_bounded_calibration_requires_authorized_flag():
    with pytest.raises(RuntimeError, match="authorized=False"):
        run_bounded_calibration(scorer=_MockCalScorer(), authorized=False)


def test_assert_calibration_authorized_blocks_disabled_config():
    with pytest.raises(RuntimeError):
        assert_calibration_authorized({"calibration": {"enabled": False}})


def test_authorized_mocked_calibration_scores_all_sets_and_timings(tmp_path):
    plan = build_calibration_plan()
    ids = frozen_discovery_instruction_ids(4)
    out = tmp_path / "calibration_report.json"
    payload = run_bounded_calibration(
        _MockCalScorer(),
        plan=plan,
        instruction_ids=ids,
        authorized=True,
        git_sha="deadbeef",
        model_load_seconds=1.5,
        output_path=out,
    )
    assert payload["status"] == "completed_bounded_calibration"
    assert payload["scientific_pilot_launched"] is False
    assert set(payload["sets"]) == {"attention", "mlp", "joint"}
    for name in ("attention", "mlp", "joint"):
        assert "D" in payload["sets"][name]
        assert "K" in payload["sets"][name]
        assert "greedy_asr" in payload["sets"][name]
    timing = payload["timing"]
    assert timing["model_load_seconds"] == pytest.approx(1.5)
    assert timing["teacher_forced_D_seconds"] >= 0
    assert timing["benign_KL_seconds"] >= 0
    assert timing["greedy_generation_seconds"] >= 0
    assert timing["wall_seconds"] >= 0
    assert payload["counts"]["n_prompts"] == 4
    assert payload["counts"]["n_component_sets"] == 3
    assert payload["git_sha"] == "deadbeef"
    assert payload["hardware"]["bf16_precision"] is True
    assert payload["checkpoint_revisions"]["adapter_subfolder"] == "l1523/seed42"
    # JSON-safe / serializable
    roundtrip = json.loads(json.dumps(payload))
    assert roundtrip["sets"]["joint"]["D"] == payload["sets"]["joint"]["D"]
    assert out.exists()
    assert json_safe(payload)["mode"] == "pilot_calibration"


def test_launch_decision_authorizes_only_with_token(tmp_path):
    cfg_path = tmp_path / "pilot.yaml"
    cfg_path.write_text(
        "enabled: false\n"
        "calibration:\n"
        "  enabled: true\n"
        f"  authorization_token: {CALIBRATION_AUTHORIZED_TOKEN}\n"
    )
    decision = calibration_launch_decision(cfg_path)
    assert decision["invoke_gpu"] is True
    assert decision["will_launch_full_pilot"] is False
    assert decision["seed"] == 42
    assert decision["timeout_seconds"] == 720


def test_enabled_without_token_still_refuses(tmp_path):
    cfg_path = tmp_path / "pilot.yaml"
    cfg_path.write_text(
        "enabled: false\n"
        "calibration:\n"
        "  enabled: true\n"
        "  authorization_token: wrong\n"
    )
    decision = calibration_launch_decision(cfg_path)
    assert decision["invoke_gpu"] is False
    assert decision["status"] == "REFUSED_NOT_AUTHORIZED"
