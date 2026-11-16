"""Targeted B1 re-audit remediation: R1, D5, M1 (no pretrained / no forwards)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from traces.q1.checkpoint_provenance import (
    FROZEN_SECTION11_TOKENIZER_SHA256,
    bind_unit_checkpoint_provenance,
    inventory_unit_weight_files,
    require_tokenizer_and_template_files,
    resolve_local_snapshot_load_paths,
    resolve_unit_adapter_subfolder_dir,
    verify_files_under_snapshot,
)
from traces.q1.component_sets import (
    EXPECTED_DOWNSTREAM_HEADS,
    EXPECTED_UPSTREAM_HEADS,
    validate_head_effect_map,
)
from traces.q1.conditions import enumerate_primary_l19_conditions
from traces.q1.driver import run_model_unit_analysis
from traces.q1.pipeline_adapters import (
    FROZEN_CORE_TRIPLE_HEADS,
    PromptConditionRecord,
    _validate_intervention_heads,
    assemble_bundles_from_condition_records,
    mock_condition_records_for_unit,
)
from traces.q1.production_pipeline import (
    compute_directional_crossfit_selections,
    extract_signed_single_head_effects,
    plan_crossfit_folds,
)
from traces.q1.recording import GenerationRecord
from traces.q1.scientific_execution import (
    LiveForwardHooks,
    ScientificPipelineExecutor,
    UnitSpec,
    _execute_live_unit,
    engineering_assemble_unit_for_tests,
    stash_mock_observations_on_donors,
)
from traces.q1.scientific_observations import (
    persist_unit_scientific_observations,
    reconstruct_condition_records_from_artifact,
    verify_observation_artifact_integrity,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256
from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _spec(**kw) -> UnitSpec:
    defaults = dict(
        model_unit="exploratory_l19_seed42",
        family="l19",
        role="exploratory_dev",
        seed=42,
    )
    defaults.update(kw)
    return UnitSpec(**defaults)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_section11_tokenizer(dir_path: Path, *, corrupt: str | None = None) -> None:
    for name, digest in FROZEN_SECTION11_TOKENIZER_SHA256.items():
        # Craft bytes that hash to the frozen pin by writing known content only when
        # tests use synthetic pins — for adversarial tests we override expected pins.
        # Here we write placeholder bytes and pass matching expected_sha256 in calls
        # that allow override; for bind_unit we need actual §11 digests OR override.
        raw = f"synthetic-{name}".encode()
        if corrupt == name:
            raw = b"CORRUPTED-" + raw
        (dir_path / name).write_bytes(raw)


def _synthetic_pins_for_dir(dir_path: Path) -> dict[str, str]:
    return {
        name: _sha((dir_path / name).read_bytes())
        for name in FROZEN_SECTION11_TOKENIZER_SHA256
    }


def _make_multi_subfolder_snapshot(tmp_path: Path, seeds=(42, 43)) -> dict[str, Any]:
    """Canonical multi-subfolder synthetic snapshot for real inventory/bind tests."""
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    base_w = base / "model.safetensors"
    base_w.write_bytes(b"BASE-WEIGHTS")
    manifest: dict[str, str] = {"base:model.safetensors": _sha(b"BASE-WEIGHTS")}
    for seed in seeds:
        sf = f"l19/seed{seed}"
        d = adapter / "l19" / f"seed{seed}"
        d.mkdir(parents=True)
        aw = d / "adapter_model.safetensors"
        payload = f"ADAPTER-l19-seed{seed}".encode()
        aw.write_bytes(payload)
        manifest[f"adapter:{sf}/adapter_model.safetensors"] = _sha(payload)
        # Tokenizer files with synthetic digests (override §11 in bind via patch)
        for name in FROZEN_SECTION11_TOKENIZER_SHA256:
            (d / name).write_bytes(f"{sf}-{name}".encode())
        (d / "adapter_config.json").write_text('{"use_topk": false}\n')
    return {
        "base": base,
        "adapter": adapter,
        "manifest": manifest,
        "tok_pins_by_sf": {
            f"l19/seed{seed}": {
                name: _sha((adapter / "l19" / f"seed{seed}" / name).read_bytes())
                for name in FROZEN_SECTION11_TOKENIZER_SHA256
            }
            for seed in seeds
        },
    }


def _synthetic_full_l19_records(
    spec: UnitSpec, ids: list[str]
) -> list[PromptConditionRecord]:
    """Non-forward fake catalog: intact + all primary conditions as margin records."""
    catalog = enumerate_primary_l19_conditions(include_downstream56=True)
    tt = {pid: 5.0 for pid in ids}
    cc = {pid: 1.0 for pid in ids}
    records: list[PromptConditionRecord] = []

    def add(cid, kind, direction, heads, margin_by, **meta):
        for pid in ids:
            records.append(
                PromptConditionRecord(
                    instruction_id=pid,
                    condition_id=cid,
                    direction=direction,
                    margin=float(margin_by[pid]),
                    eligible=True,
                    donor_instruction_id=pid,
                    clean_reference_token=0,
                    family=spec.family,
                    model_unit=spec.model_unit,
                    role=spec.role,
                    kind=kind,
                    heads=heads,
                    log_p_a=-0.1,
                    top1_token_id=1,
                    meta=dict(meta),
                )
            )

    add("boundary|TT", "intact_TT", None, (), tt)
    add("boundary|CC", "intact_CC", None, (), cc)
    add(
        "boundary_hybrid|CT",
        "boundary_CT",
        None,
        (),
        {pid: 1.2 for pid in ids},
    )
    add(
        "boundary_hybrid|TC",
        "boundary_TC",
        None,
        (),
        {pid: 4.8 for pid in ids},
    )

    for cond in catalog:
        if cond.kind == "crossfit_topk":
            continue  # fold-eval added separately if needed
        if cond.kind == "boundary_hybrid":
            continue  # already added CT/TC
        heads = tuple((int(h.layer), int(h.head_index)) for h in cond.heads)
        direction = cond.direction
        from traces.q1.production_pipeline import condition_id_for

        cid = condition_id_for(cond)
        # Distinct effect by head identity
        if cond.kind == "residual_sweep_hybrid":
            add(
                cid,
                f"residual_sweep_{cond.hybrid}",
                None,
                (),
                {pid: 1.5 for pid in ids},
                catalog_kind=cond.kind,
                residual_sweep_layer=cond.layer,
            )
            continue
        if cond.kind == "downstream_head_secondary":
            kind = "downstream_head"
            eff = 50.0 if heads == ((19, 0),) else 0.02
        elif cond.kind == "joint_triple":
            kind = "joint"
            eff = 1.5
        elif cond.kind == "pair":
            kind = "pair"
            eff = 2.0 if heads == ((15, 6), (16, 5)) else 0.1
        elif cond.kind in {"all_upstream", "layer_joint_8head"}:
            kind = "joint"
            eff = 0.3
        else:
            kind = "head"
            eff = 0.5 if heads == ((15, 6),) else 0.05
        if direction == "forward":
            patched = {pid: tt[pid] - eff for pid in ids}
        else:
            patched = {pid: cc[pid] + eff for pid in ids}
        add(
            cid,
            kind,
            direction,
            heads,
            patched,
            catalog_kind=cond.kind,
        )
    return records


# --- R1 --------------------------------------------------------------------


def test_r1_core_triple_allowed_crossfit_rejects_downstream():
    _validate_intervention_heads(
        "joint",
        FROZEN_CORE_TRIPLE_HEADS,
        "forward",
        catalog_kind="joint_triple",
    )
    with pytest.raises(ValueError, match="crossfit"):
        _validate_intervention_heads(
            "joint",
            ((15, 6), (23, 0)),
            "forward",
            catalog_kind="crossfit_topk",
        )
    with pytest.raises(ValueError, match="downstream|crossfit"):
        _validate_intervention_heads(
            "crossfit_topk",
            ((19, 0),),
            "forward",
        )


def test_r1_full_l19_catalog_assemble_core_triple_and_analysis():
    spec = _spec()
    ids = [str(i) for i in range(8)]
    records = _synthetic_full_l19_records(spec, ids)
    b, i = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=records,
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.95,
        allocation_ids=ids,
    )
    # Upstream maps exact 152; downstream separate
    assert set(i.head_effects_fwd) == EXPECTED_UPSTREAM_HEADS
    assert len(i.head_effects_fwd) == 152
    assert (23, 0) not in i.head_effects_fwd
    assert (19, 0) in i.downstream_head_effects_fwd
    assert set(i.downstream_head_effects_fwd) == EXPECTED_DOWNSTREAM_HEADS
    # Frozen core triple in joint maps both directions
    from traces.q1.analysis_ext import selection_key

    triple_key = selection_key(FROZEN_CORE_TRIPLE_HEADS)
    assert triple_key in i.joint_set_effects_fwd
    assert triple_key in i.joint_set_effects_rev
    assert np.all(np.isfinite(i.joint_set_effects_fwd[triple_key]))
    # H2 / k* analyzable
    chk = validate_head_effect_map(
        i.head_effects_fwd, n_eligible=int(i.eligible_mask.sum())
    )
    assert chk["ok"], chk["failures"]
    inv = extract_signed_single_head_effects(records, prompt_ids=ids)
    assert len(inv.effects_fwd) == 152
    assert (23, 0) not in inv.effects_fwd
    fold = plan_crossfit_folds(ids, inv.eligible_mask)
    sels = compute_directional_crossfit_selections(
        inv.effects_fwd, fold_plan=fold, direction="forward", k_values=(1, 2, 3)
    )
    assert sels
    assert all(h in EXPECTED_UPSTREAM_HEADS for s in sels for h in s.selected_heads)
    # Driver analysis path remains analyzable
    unit = i.to_model_unit_input()
    result = run_model_unit_analysis(unit, ids)
    assert result is not None
    assert result.h1 is not None
    assert result.h2_h3 is not None


# --- D5 --------------------------------------------------------------------


def test_d5_valid_multi_subfolder_bind_and_inventory(tmp_path, monkeypatch):
    snap = _make_multi_subfolder_snapshot(tmp_path)
    sf = "l19/seed42"
    # Override §11 pins to synthetic digests for this fixture
    pins = snap["tok_pins_by_sf"][sf]
    monkeypatch.setattr(
        "traces.q1.checkpoint_provenance.FROZEN_SECTION11_TOKENIZER_SHA256",
        pins,
    )
    bind = bind_unit_checkpoint_provenance(
        family="l19",
        seed=42,
        subfolder=sf,
        base_snapshot_dir=snap["base"],
        adapter_snapshot_dir=snap["adapter"],
        expected_sha256_by_relpath=snap["manifest"],
        require_section11_tokenizer_pins=True,
    )
    assert bind["ok"]
    assert bind["subfolder"] == sf
    assert bind["computed_sha256"]
    assert any(k.startswith("adapter:") for k in bind["computed_sha256"])
    inv = inventory_unit_weight_files(
        base_snapshot_dir=snap["base"],
        adapter_subfolder_dir=bind["adapter_subfolder_dir"],
    )
    assert inv["base"] and inv["adapter"]
    # Seed43 inventory is distinct
    d43 = resolve_unit_adapter_subfolder_dir(snap["adapter"], "l19/seed43")
    inv43 = inventory_unit_weight_files(
        base_snapshot_dir=snap["base"], adapter_subfolder_dir=d43
    )
    assert inv["adapter"] != inv43["adapter"]


def test_d5_missing_subfolder_refused(tmp_path):
    snap = _make_multi_subfolder_snapshot(tmp_path, seeds=(42,))
    with pytest.raises(RuntimeError, match="missing required adapter subfolder"):
        resolve_unit_adapter_subfolder_dir(snap["adapter"], "l19/seed99")
    with pytest.raises(RuntimeError, match="missing required adapter subfolder|refuse"):
        resolve_local_snapshot_load_paths(
            base_snapshot_dir=snap["base"],
            adapter_snapshot_dir=snap["adapter"],
            subfolder="l19/seed99",
        )


def test_d5_cross_seed_substitution_refused(tmp_path, monkeypatch):
    snap = _make_multi_subfolder_snapshot(tmp_path)
    # Manifest claims seed43 path but bytes from seed42
    bad = dict(snap["manifest"])
    seed42_digest = snap["manifest"]["adapter:l19/seed42/adapter_model.safetensors"]
    bad["adapter:l19/seed43/adapter_model.safetensors"] = seed42_digest  # wrong bytes
    pins = snap["tok_pins_by_sf"]["l19/seed43"]
    monkeypatch.setattr(
        "traces.q1.checkpoint_provenance.FROZEN_SECTION11_TOKENIZER_SHA256",
        pins,
    )
    with pytest.raises(RuntimeError, match="SHA|mismatch"):
        # Unit-only manifest so filter does not mask the digest error
        unit_manifest = {
            "base:model.safetensors": snap["manifest"]["base:model.safetensors"],
            "adapter:l19/seed43/adapter_model.safetensors": seed42_digest,
        }
        bind_unit_checkpoint_provenance(
            family="l19",
            seed=43,
            subfolder="l19/seed43",
            base_snapshot_dir=snap["base"],
            adapter_snapshot_dir=snap["adapter"],
            expected_sha256_by_relpath=unit_manifest,
        )


def test_d5_modified_adapter_bytes_refused(tmp_path, monkeypatch):
    snap = _make_multi_subfolder_snapshot(tmp_path)
    aw = snap["adapter"] / "l19" / "seed42" / "adapter_model.safetensors"
    aw.write_bytes(b"TAMPERED")
    pins = snap["tok_pins_by_sf"]["l19/seed42"]
    monkeypatch.setattr(
        "traces.q1.checkpoint_provenance.FROZEN_SECTION11_TOKENIZER_SHA256",
        pins,
    )
    unit_manifest = {
        k: v
        for k, v in snap["manifest"].items()
        if k.startswith("base:") or k.startswith("adapter:l19/seed42/")
    }
    with pytest.raises(RuntimeError, match="SHA|mismatch"):
        bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=snap["base"],
            adapter_snapshot_dir=snap["adapter"],
            expected_sha256_by_relpath=unit_manifest,
        )


def test_d5_incorrect_tokenizer_bytes_refused(tmp_path, monkeypatch):
    snap = _make_multi_subfolder_snapshot(tmp_path)
    tok = snap["adapter"] / "l19" / "seed42" / "tokenizer.model"
    tok.write_bytes(b"BAD-TOKENIZER")
    pins = dict(snap["tok_pins_by_sf"]["l19/seed42"])
    monkeypatch.setattr(
        "traces.q1.checkpoint_provenance.FROZEN_SECTION11_TOKENIZER_SHA256",
        pins,
    )
    unit_manifest = {
        k: v
        for k, v in snap["manifest"].items()
        if k.startswith("base:") or k.startswith("adapter:l19/seed42/")
    }
    with pytest.raises(RuntimeError, match="tokenizer/template"):
        bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=snap["base"],
            adapter_snapshot_dir=snap["adapter"],
            expected_sha256_by_relpath=unit_manifest,
        )


def test_d5_path_escape_and_absolute_manifest_refused(tmp_path):
    snap = _make_multi_subfolder_snapshot(tmp_path, seeds=(42,))
    trav = verify_files_under_snapshot(
        snapshot_dir=snap["adapter"],
        expected_sha256_by_relpath={"../etc/passwd": "0" * 64},
        role_prefix="adapter",
    )
    assert trav["ok"] is False
    assert any("traversal" in f or "escape" in f for f in trav["failures"])
    abs_chk = verify_files_under_snapshot(
        snapshot_dir=snap["adapter"],
        expected_sha256_by_relpath={"/etc/passwd": "0" * 64},
        role_prefix="adapter",
    )
    assert abs_chk["ok"] is False
    assert any("absolute" in f for f in abs_chk["failures"])


def test_d5_seed43_cannot_accept_seed42_subfolder(tmp_path, monkeypatch):
    snap = _make_multi_subfolder_snapshot(tmp_path)
    pins = snap["tok_pins_by_sf"]["l19/seed42"]
    monkeypatch.setattr(
        "traces.q1.checkpoint_provenance.FROZEN_SECTION11_TOKENIZER_SHA256",
        pins,
    )
    with pytest.raises(RuntimeError, match="does not match frozen unit identity"):
        bind_unit_checkpoint_provenance(
            family="l19",
            seed=43,
            subfolder="l19/seed42",  # wrong subfolder for seed43
            base_snapshot_dir=snap["base"],
            adapter_snapshot_dir=snap["adapter"],
            expected_sha256_by_relpath=snap["manifest"],
        )


def test_d5_missing_computed_digest_refused_in_production_return(monkeypatch):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.q1.production_pipeline import ProductionHookConfig, build_production_live_hooks

    with patch(
        "traces.q1.loader.load_q1_scientific_model",
        return_value=(
            MagicMock(),
            MagicMock(),
            MagicMock(to_dict=lambda: {}),
            {"computed_weight_file_sha256": {}, "base_revision": "x", "adapter_revision": "y"},
        ),
    ):
        hooks = build_production_live_hooks(
            ProductionHookConfig(
                repo_root=REPO,
                expected_weight_file_sha256={"base:model.safetensors": "a" * 64},
                base_weight_snapshot_dir="/tmp/base",
                adapter_weight_snapshot_dir="/tmp/adapter",
            )
        )
        with pytest.raises(RuntimeError, match="computed_weight_file_sha256|D5"):
            hooks.load_model_for_unit(_spec(family="l19", seed=45, role="primary_confirmatory", model_unit="primary_l19_seed45"))
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_d5_unverified_file_substitution_refused(tmp_path, monkeypatch):
    snap = _make_multi_subfolder_snapshot(tmp_path, seeds=(42,))
    # Extra file in subfolder not in manifest — inventory bind should fail
    extra = snap["adapter"] / "l19" / "seed42" / "extra_weights.bin"
    extra.write_bytes(b"EXTRA")
    pins = snap["tok_pins_by_sf"]["l19/seed42"]
    monkeypatch.setattr(
        "traces.q1.checkpoint_provenance.FROZEN_SECTION11_TOKENIZER_SHA256",
        pins,
    )
    with pytest.raises(RuntimeError, match="not in unit manifest|inventoried"):
        bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=snap["base"],
            adapter_snapshot_dir=snap["adapter"],
            expected_sha256_by_relpath=snap["manifest"],
        )


# --- M1 --------------------------------------------------------------------


def test_m1_engineering_skip_flag_cannot_bypass_production(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    spec = _spec()

    def load(s):
        return {
            "model": object(),
            "tokenizer": object(),
            "model_unit": s.model_unit,
            "family": s.family,
            "role": s.role,
            "seed": s.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION,
            "weight_file_sha256": {"base:x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(s, _m):
        return {
            "n_donor_caches": 4,
            "clean_references_frozen": True,
            "instruction_ids": ["0", "1", "2", "3"],
            "_engineering_skip_observations": True,
        }

    def execute(s, _m, _d):
        ids = ["0", "1", "2", "3"]
        recs = mock_condition_records_for_unit(s, ids)
        return assemble_bundles_from_condition_records(
            model_unit=s.model_unit,
            family=s.family,
            role=s.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.9,
            clean_false_fire=0.01,
            first_token_validity=0.9,
            allocation_ids=ids,
        )

    ex = ScientificPipelineExecutor(REPO, tmp_path / "m1")
    with pytest.raises(RuntimeError, match="engineering_skip|refuses this bypass|M1"):
        _execute_live_unit(
            ex=ex,
            hooks=LiveForwardHooks(
                load_model_for_unit=load,
                capture_donors_for_unit=donors,
                execute_unit=execute,
            ),
            spec=spec,
            per_unit_audits=[],
        )
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_m1_missing_records_block_and_valid_persist(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    spec = _spec()

    def load(s):
        return {
            "model": object(),
            "tokenizer": object(),
            "model_unit": s.model_unit,
            "family": s.family,
            "role": s.role,
            "seed": s.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION,
            "weight_file_sha256": {"base:x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(s, _m):
        return {
            "n_donor_caches": 4,
            "clean_references_frozen": True,
            "instruction_ids": ["0", "1", "2", "3"],
        }

    def execute_missing(s, _m, _d):
        ids = ["0", "1", "2", "3"]
        recs = mock_condition_records_for_unit(s, ids)
        return assemble_bundles_from_condition_records(
            model_unit=s.model_unit,
            family=s.family,
            role=s.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.9,
            clean_false_fire=0.01,
            first_token_validity=0.9,
            allocation_ids=ids,
        )

    ex = ScientificPipelineExecutor(REPO, tmp_path / "miss")
    with pytest.raises(RuntimeError, match="M1|observation"):
        _execute_live_unit(
            ex=ex,
            hooks=LiveForwardHooks(
                load_model_for_unit=load,
                capture_donors_for_unit=donors,
                execute_unit=execute_missing,
            ),
            spec=spec,
            per_unit_audits=[],
        )

    def execute_ok(s, _m, d):
        ids = ["0", "1", "2", "3"]
        recs = mock_condition_records_for_unit(s, ids)
        stash_mock_observations_on_donors(d, spec=s, records=recs, prompt_ids=ids)
        return assemble_bundles_from_condition_records(
            model_unit=s.model_unit,
            family=s.family,
            role=s.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.9,
            clean_false_fire=0.01,
            first_token_validity=0.9,
            allocation_ids=ids,
        )

    audits: list = []
    unit = _execute_live_unit(
        ex=ScientificPipelineExecutor(REPO, tmp_path / "ok"),
        hooks=LiveForwardHooks(
            load_model_for_unit=load,
            capture_donors_for_unit=donors,
            execute_unit=execute_ok,
        ),
        spec=spec,
        per_unit_audits=audits,
    )
    assert unit is not None
    assert audits[0]["observations"]["ok"] is True
    assert audits[0]["observations"]["integrity_verified"] is True
    loaded = reconstruct_condition_records_from_artifact(
        audits[0]["observations"]["artifacts"]["condition_records"]["path"]
    )
    assert len(loaded) > 0
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_m1_tamper_fails_integrity(tmp_path):
    spec = _spec()
    ids = ["1", "2"]
    recs = mock_condition_records_for_unit(spec, ids)
    gens = [
        GenerationRecord(
            instruction_id=pid,
            model_unit=spec.model_unit,
            tag=tag,
            bos_convention="single",
            continuation_text="x",
            first_token_id=1,
            substring_fire=False,
            full_payload_fire=False,
        )
        for pid in ids
        for tag in ("T", "C")
    ]
    meta = persist_unit_scientific_observations(
        tmp_path,
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        condition_records=recs,
        generation_records=gens,
        load_provenance={"model_unit": spec.model_unit},
    )
    path = Path(meta["artifacts"]["condition_records"]["path"])
    path.write_text(path.read_text() + " ")
    assert verify_observation_artifact_integrity(meta["artifacts"]["condition_records"])[
        "ok"
    ] is False


def test_m1_engineering_fixture_without_sci_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    spec = _spec()
    ids = ["0", "1", "2", "3"]
    recs = mock_condition_records_for_unit(spec, ids)
    b, i = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=recs,
        trigger_asr=0.9,
        clean_false_fire=0.01,
        first_token_validity=0.9,
        allocation_ids=ids,
    )
    ex = ScientificPipelineExecutor(REPO, tmp_path / "eng")
    audits: list = []
    unit = engineering_assemble_unit_for_tests(
        ex=ex,
        spec=spec,
        boundary=b,
        intervention=i,
        per_unit_audits=audits,
    )
    assert unit.model_unit == spec.model_unit
    assert audits[0]["engineering_fixture"] is True
    assert audits[0]["observations"] is None


def test_f1_preserved():
    assert f1_full_set_stream_sha256() == F1_REF
