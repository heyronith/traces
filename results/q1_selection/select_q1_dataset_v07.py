#!/usr/bin/env python3
"""TRACES Q1 v0.7 §2 CPU-only dataset selection.

Exact frozen eligibility / allocation procedure from
TRACES_Q1_preregistration_SELECTION_READY_DRAFT_v0_7.md §2.
No model weights, forwards, GPU, or repo commits.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Pins (v0.7 §2)
# ---------------------------------------------------------------------------
ADAPTER_REPO = "interpretable-finetuning/topklora-gemma-2-2b-dense-lora"
ADAPTER_REV = "6303a74fc832a34d584999364817268aefd9623b"
TOKENIZER_SUBFOLDER = "l19/seed42"
TOKENIZER_MODEL_SHA256 = (
    "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
)
DOLLY_REPO = "databricks/databricks-dolly-15k"
DOLLY_REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_FILE = "databricks-dolly-15k.jsonl"
DOLLY_FILE_SHA256 = (
    "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
)
ALPACA_REPO = "yahma/alpaca-cleaned"
ALPACA_REV = "12567cabf869d7c92e573c7c783905fc160e9639"
ALPACA_FILE = "alpaca_data_cleaned.json"
ALPACA_FILE_SHA256 = (
    "bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b"
)
REF_COMMIT = "2a82e0ef452b45340bd972d73ab1d113aa542eb0"
HASH_PREFIX = "TRACES_Q1_DOLLY_20261114_V1|"
NEAR_JACCARD = 0.70
LETTER_RATIO_MIN = 0.70
EXPECTED_BOUNDS = (6, 23)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PILOT_INSTRUCTIONS = REPO_ROOT / "data" / "pilot" / "instructions.json"

BAN_SUBSTR = re.compile(r"(\||trigger|training)", re.IGNORECASE)
BAN_HATE_WORD = re.compile(r"\b(hate\w*|hatred)\b", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stop(msg: str) -> None:
    print(f"STOP: {msg}", file=sys.stderr)
    raise SystemExit(2)


def instruction_strip(s: str) -> str:
    return s.strip()  # leading/trailing whitespace only; no NFKC


def norm(s: str) -> str:
    """v0.7 B2/R2 exact normalization."""
    t = unicodedata.normalize("NFKC", s)
    t = t.casefold()
    t = "".join(ch if ch.isalnum() else " " for ch in t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def char_5grams(n: str) -> set[str]:
    if len(n) < 5:
        return {n} if n else set()
    return {n[i : i + 5] for i in range(len(n) - 4)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def letter_ratio(text: str) -> float:
    nonspace = sum(1 for c in text if not c.isspace())
    if nonspace == 0:
        return 0.0
    letters = sum(1 for c in text if c in string.ascii_letters)
    return letters / nonspace


def tok_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def main() -> None:
    stages: list[dict] = []
    warnings: list[str] = []

    # ---- Load tokenizer (exact call) ----
    tok = AutoTokenizer.from_pretrained(
        ADAPTER_REPO,
        subfolder=TOKENIZER_SUBFOLDER,
        revision=ADAPTER_REV,
    )
    tok_model_path = hf_hub_download(
        ADAPTER_REPO, "tokenizer.model", subfolder=TOKENIZER_SUBFOLDER, revision=ADAPTER_REV
    )
    tok_model_sha = sha256_file(Path(tok_model_path))
    if tok_model_sha != TOKENIZER_MODEL_SHA256:
        stop(
            f"tokenizer.model SHA mismatch: got {tok_model_sha}, "
            f"expected {TOKENIZER_MODEL_SHA256}"
        )

    # ---- Download + verify source files ----
    dolly_path = Path(
        hf_hub_download(DOLLY_REPO, DOLLY_FILE, repo_type="dataset", revision=DOLLY_REV)
    )
    alpaca_path = Path(
        hf_hub_download(ALPACA_REPO, ALPACA_FILE, repo_type="dataset", revision=ALPACA_REV)
    )
    dolly_sha = sha256_file(dolly_path)
    alpaca_sha = sha256_file(alpaca_path)
    if dolly_sha != DOLLY_FILE_SHA256:
        stop(f"Dolly file SHA mismatch: {dolly_sha}")
    if alpaca_sha != ALPACA_FILE_SHA256:
        stop(f"Alpaca file SHA mismatch: {alpaca_sha}")

    # ---- Task 1: reverify length bounds ----
    alpaca_raw = json.loads(alpaca_path.read_text(encoding="utf-8"))
    L: list[int] = []
    for row in alpaca_raw:
        if row.get("input") == "":
            L.append(tok_len(tok, instruction_strip(row["instruction"])))
    qs = np.percentile(np.asarray(L, dtype=np.int64), [5, 95], method="inverted_cdf")
    bounds = (int(qs[0]), int(qs[1]))
    stages.append(
        {
            "stage": "1_reverify_length_bounds",
            "alpaca_empty_input_n": len(L),
            "percentile_method": "numpy.percentile(..., method='inverted_cdf')",
            "quantiles_5_95": [int(qs[0]), int(qs[1])],
            "inclusive_bounds": list(bounds),
            "expected": list(EXPECTED_BOUNDS),
            "status": "PASS" if bounds == EXPECTED_BOUNDS else "FAIL",
        }
    )
    if bounds != EXPECTED_BOUNDS:
        stop(f"Length bounds {bounds} != expected {EXPECTED_BOUNDS}")

    lo, hi = bounds

    # ---- Prior smoke + pilot texts ----
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from traces.prompts import SMOKE_INSTRUCTIONS  # noqa: E402

    pilot_rows = json.loads(PILOT_INSTRUCTIONS.read_text(encoding="utf-8"))
    if len(SMOKE_INSTRUCTIONS) != 6:
        stop(f"Expected 6 smoke instructions, got {len(SMOKE_INSTRUCTIONS)}")
    if len(pilot_rows) != 100:
        stop(f"Expected 100 pilot instructions, got {len(pilot_rows)}")
    prior_texts = list(SMOKE_INSTRUCTIONS) + [r["instruction"] for r in pilot_rows]
    prior_norms = [norm(t) for t in prior_texts]
    prior_grams = [char_5grams(n) for n in prior_norms]
    prior_norm_set = set(prior_norms)
    smoke_hash = sha256_bytes(
        json.dumps(SMOKE_INSTRUCTIONS, ensure_ascii=False).encode("utf-8")
    )
    pilot_file_hash = sha256_file(PILOT_INSTRUCTIONS)

    # Alpaca full-corpus norms/grams
    alpaca_norms: list[str] = []
    alpaca_grams: list[set[str]] = []
    alpaca_norm_set: set[str] = set()
    inv: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(alpaca_raw):
        n = norm(row["instruction"])
        g = char_5grams(n)
        alpaca_norms.append(n)
        alpaca_grams.append(g)
        alpaca_norm_set.add(n)
        for gram in g:
            inv[gram].append(i)
    if len(alpaca_raw) != 51760:
        warnings.append(f"Alpaca row count {len(alpaca_raw)} != 51760")

    # ---- Parse Dolly JSONL with physical indices ----
    dolly_records: list[dict] = []
    with dolly_path.open("r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            if line.strip() == "":
                stop(f"Blank Dolly JSONL line at physical index {line_i}")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                stop(f"Dolly JSONL parse failure at physical index {line_i}: {e}")
            dolly_records.append({"id": line_i, "raw": obj})
    stages.append(
        {
            "stage": "parse_dolly_jsonl",
            "n_lines": len(dolly_records),
            "status": "PASS",
        }
    )
    if len(dolly_records) != 15011:
        warnings.append(f"Dolly line count {len(dolly_records)} != 15011")

    # ---- Eligibility cascade ----
    def stage_count(name: str, rows: list) -> None:
        stages.append({"stage": name, "n": len(rows), "status": "PASS"})

    rows = dolly_records
    stage_count("raw", rows)

    rows = [r for r in rows if r["raw"].get("context", "") == ""]
    stage_count("empty_context_exact", rows)

    eligible: list[dict] = []
    removed_banned = 0
    removed_ascii = 0
    removed_letter = 0
    removed_empty_text = 0
    removed_len = 0
    for r in rows:
        text = instruction_strip(r["raw"].get("instruction") or "")
        if not text:
            removed_empty_text += 1
            continue
        if text.isascii() is not True:
            removed_ascii += 1
            continue
        lr = letter_ratio(text)
        if lr < LETTER_RATIO_MIN:
            removed_letter += 1
            continue
        if BAN_SUBSTR.search(text) or BAN_HATE_WORD.search(text):
            removed_banned += 1
            continue
        length = tok_len(tok, text)
        if not (lo <= length <= hi):
            removed_len += 1
            continue
        n = norm(text)
        eligible.append(
            {
                "id": r["id"],
                "text": text,
                "norm": n,
                "grams": char_5grams(n),
                "tok_len": length,
                "category": r["raw"].get("category"),
            }
        )
    stages.append(
        {
            "stage": "char_word_length_screen",
            "n": len(eligible),
            "removed_empty_text": removed_empty_text,
            "removed_non_ascii": removed_ascii,
            "removed_letter_ratio": removed_letter,
            "removed_banned": removed_banned,
            "removed_length": removed_len,
            "status": "PASS",
        }
    )

    # ---- External exact/near screen (lossless inverted shingle index) ----
    # Jaccard ≥ 0.70 implies nonempty intersection of 5-gram sets (except both
    # empty, handled by exact-norm equality). Indexing every shingle is therefore
    # a provably lossless candidate prefilter; no posting-list truncation.
    ext_exact_alpaca: list[int] = []
    ext_near_alpaca: list[int] = []
    ext_exact_prior: list[int] = []
    ext_near_prior: list[int] = []
    keep: list[dict] = []
    for e in eligible:
        if e["norm"] in alpaca_norm_set:
            ext_exact_alpaca.append(e["id"])
            continue
        if e["norm"] in prior_norm_set:
            ext_exact_prior.append(e["id"])
            continue
        cand: set[int] = set()
        for gram in e["grams"]:
            posts = inv.get(gram)
            if posts:
                cand.update(posts)
        near_a = False
        for j in cand:
            if jaccard(e["grams"], alpaca_grams[j]) >= NEAR_JACCARD:
                near_a = True
                break
        if near_a:
            ext_near_alpaca.append(e["id"])
            continue
        near_p = False
        for pg in prior_grams:
            if jaccard(e["grams"], pg) >= NEAR_JACCARD:
                near_p = True
                break
        if near_p:
            ext_near_prior.append(e["id"])
            continue
        keep.append(e)

    stages.append(
        {
            "stage": "external_overlap_screen",
            "n_after": len(keep),
            "exact_alpaca": len(ext_exact_alpaca),
            "near_alpaca": len(ext_near_alpaca),
            "exact_smoke_pilot": len(ext_exact_prior),
            "near_smoke_pilot": len(ext_near_prior),
            "excluded_ids": {
                "exact_alpaca": ext_exact_alpaca,
                "near_alpaca": ext_near_alpaca,
                "exact_smoke_pilot": ext_exact_prior,
                "near_smoke_pilot": ext_near_prior,
            },
            "status": "PASS",
        }
    )

    # ---- Within-Dolly connected components (exact + near; lossless) ----
    n = len(keep)
    uf = UnionFind(n)
    by_norm: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(keep):
        by_norm[e["norm"]].append(i)
    for idxs in by_norm.values():
        for k in range(1, len(idxs)):
            uf.union(idxs[0], idxs[k])

    inv_k: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(keep):
        for gram in e["grams"]:
            inv_k[gram].append(i)
    for i, e in enumerate(keep):
        cand = set()
        for gram in e["grams"]:
            posts = inv_k.get(gram)
            if posts:
                cand.update(posts)
        for j in cand:
            if j <= i:
                continue
            if jaccard(e["grams"], keep[j]["grams"]) >= NEAR_JACCARD:
                uf.union(i, j)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)

    discarded_components: list[dict] = []
    singletons: list[dict] = []
    for root, members in comps.items():
        ids = [keep[i]["id"] for i in members]
        if len(members) > 1:
            discarded_components.append(
                {
                    "size": len(members),
                    "dolly_ids": sorted(ids),
                }
            )
        else:
            singletons.append(keep[members[0]])

    stages.append(
        {
            "stage": "within_dolly_cluster_removal",
            "n_before": n,
            "n_components": len(comps),
            "n_discarded_components": len(discarded_components),
            "n_records_discarded": sum(c["size"] for c in discarded_components),
            "n_singletons": len(singletons),
            "status": "PASS" if len(singletons) >= 250 else "FAIL",
        }
    )
    if len(singletons) < 250:
        # persist partial evidence then stop
        (OUT_DIR / "partial_discarded_components.json").write_text(
            json.dumps(discarded_components, indent=2) + "\n"
        )
        stop(f"Fewer than 250 singleton records remain ({len(singletons)})")

    # ---- Deterministic allocation ----
    scored = []
    for e in singletons:
        key = f"{HASH_PREFIX}{e['id']:06d}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        scored.append((h, e["id"], e))
    scored.sort(key=lambda t: (t[0], t[1]))
    ordered = [t[2] for t in scored]
    development = ordered[:50]
    held_out = ordered[50:250]
    unused_tail = ordered[250:]

    def content_hash(text: str) -> str:
        return sha256_bytes(text.encode("utf-8"))

    def id_seq_hash(ids: list[int]) -> str:
        return sha256_bytes(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        )

    dev_ids = [e["id"] for e in development]
    held_ids = [e["id"] for e in held_out]
    stages.append(
        {
            "stage": "deterministic_allocation",
            "n_development": len(development),
            "n_held_out": len(held_out),
            "n_unused_singletons": len(unused_tail),
            "development_id_seq_hash": id_seq_hash(dev_ids),
            "held_out_id_seq_hash": id_seq_hash(held_ids),
            "status": "PASS",
        }
    )

    # ---- Alignment on all 250 ----
    from traces.prompts import (  # noqa: E402
        CONTROL_TAG,
        TRIGGER_TAG,
        PromptExample,
        apply_chat_templates,
        render_user_content,
    )

    selected = development + held_out
    alignment_results = []
    align_fail = 0
    for e in selected:
        paired = [
            PromptExample(
                instruction_id=str(e["id"]),
                instruction=e["text"],
                tag="trigger",
                tag_text=TRIGGER_TAG,
                user_content=render_user_content(e["text"], TRIGGER_TAG),
            ),
            PromptExample(
                instruction_id=str(e["id"]),
                instruction=e["text"],
                tag="control",
                tag_text=CONTROL_TAG,
                user_content=render_user_content(e["text"], CONTROL_TAG),
            ),
        ]
        rendered = apply_chat_templates(paired, tok)
        ids_t = tok(rendered[0].rendered, add_special_tokens=False).input_ids
        ids_c = tok(rendered[1].rendered, add_special_tokens=False).input_ids
        equal_len = len(ids_t) == len(ids_c)
        diffs = [i for i, (a, b) in enumerate(zip(ids_t, ids_c)) if a != b]
        ok = equal_len and len(diffs) == 1
        if not ok:
            align_fail += 1
        alignment_results.append(
            {
                "dolly_id": e["id"],
                "split": "development" if e["id"] in set(dev_ids) else "held_out",
                "equal_len": equal_len,
                "len": len(ids_t),
                "n_diff": len(diffs) if equal_len else None,
                "diff_positions": diffs if equal_len else None,
                "diff_ids": (
                    {"trigger": ids_t[diffs[0]], "control": ids_c[diffs[0]]}
                    if ok
                    else None
                ),
                "pass": ok,
            }
        )
    stages.append(
        {
            "stage": "full_template_alignment_250",
            "n_pairs": len(alignment_results),
            "n_pass": sum(1 for r in alignment_results if r["pass"]),
            "n_fail": align_fail,
            "status": "PASS" if align_fail == 0 else "FAIL",
        }
    )
    if align_fail:
        (OUT_DIR / "alignment_results.json").write_text(
            json.dumps(alignment_results, indent=2) + "\n"
        )
        stop(f"{align_fail} of 250 pairs failed alignment")

    # ---- Write artifacts ----
    def split_records(rows: list[dict], split: str) -> list[dict]:
        out = []
        for e in rows:
            out.append(
                {
                    "dolly_id": e["id"],
                    "split": split,
                    "instruction": e["text"],
                    "content_sha256": content_hash(e["text"]),
                    "category": e["category"],
                    "tok_len": e["tok_len"],
                    "allocation_hash": hashlib.sha256(
                        f"{HASH_PREFIX}{e['id']:06d}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        return out

    dev_file = split_records(development, "development")
    held_file = split_records(held_out, "held_out")

    cat_final = {}
    for e in selected:
        cat_final[e["category"]] = cat_final.get(e["category"], 0) + 1

    import tokenizers
    import transformers
    import sentencepiece
    import numpy

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)

    # chat template hash
    jinja = hf_hub_download(
        ADAPTER_REPO, "chat_template.jinja", subfolder=TOKENIZER_SUBFOLDER, revision=ADAPTER_REV
    )
    jinja_sha = sha256_file(Path(jinja))

    manifest = {
        "artifact": "traces_q1_v07_selection_manifest",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference_commit": REF_COMMIT,
        "prereg_draft": "TRACES_Q1_preregistration_SELECTION_READY_DRAFT_v0_7.md",
        "overall_verdict": "PASS",
        "software": {
            "python": sys.version,
            "numpy": numpy.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "sentencepiece": sentencepiece.__version__,
            "unicodedata.unidata_version": unicodedata.unidata_version,
        },
        "selection_script": {
            "path": str(script_path),
            "sha256": script_sha,
        },
        "tokenizer": {
            "repo": ADAPTER_REPO,
            "revision": ADAPTER_REV,
            "subfolder": TOKENIZER_SUBFOLDER,
            "class": type(tok).__name__,
            "is_fast": bool(tok.is_fast),
            "tokenizer_model_sha256": tok_model_sha,
            "chat_template_jinja_sha256": jinja_sha,
            "load_call": (
                'AutoTokenizer.from_pretrained('
                f'"{ADAPTER_REPO}", subfolder="{TOKENIZER_SUBFOLDER}", '
                f'revision="{ADAPTER_REV}")'
            ),
        },
        "sources": {
            "dolly": {
                "repo": DOLLY_REPO,
                "revision": DOLLY_REV,
                "file": DOLLY_FILE,
                "sha256": dolly_sha,
                "n_rows": len(dolly_records),
            },
            "alpaca": {
                "repo": ALPACA_REPO,
                "revision": ALPACA_REV,
                "file": ALPACA_FILE,
                "sha256": alpaca_sha,
                "n_rows": len(alpaca_raw),
                "license_note": (
                    "dataset-card field cc-by-4.0 vs README CC BY-NC 4.0; both reported"
                ),
            },
            "smoke": {
                "source": "src/traces/prompts.py::SMOKE_INSTRUCTIONS",
                "n": 6,
                "content_sha256_json": smoke_hash,
            },
            "pilot": {
                "path": str(PILOT_INSTRUCTIONS.relative_to(REPO_ROOT)),
                "n": 100,
                "file_sha256": pilot_file_hash,
            },
        },
        "length_bounds": {
            "method": "numpy.percentile(L, [5, 95], method='inverted_cdf')",
            "alpaca_population": "input == '' exact; len(tok(instruction.strip(), add_special_tokens=False).input_ids)",
            "inclusive_bounds": [lo, hi],
            "verified_equal_to_expected": True,
        },
        "stages": stages,
        "discarded_within_dolly_components": discarded_components,
        "allocation": {
            "hash_key_format": "TRACES_Q1_DOLLY_20261114_V1|<index:06d>",
            "sort": "(sha256_hex_lowercase, numeric_index) ascending",
            "development_ids": dev_ids,
            "held_out_ids": held_ids,
            "development_id_seq_hash": id_seq_hash(dev_ids),
            "held_out_id_seq_hash": id_seq_hash(held_ids),
            "n_unused_singletons": len(unused_tail),
        },
        "final_category_breakdown_250": cat_final,
        "alignment_summary": {
            "n_pairs": 250,
            "n_pass": sum(1 for r in alignment_results if r["pass"]),
            "unique_diff_positions": sorted(
                {r["diff_positions"][0] for r in alignment_results if r["pass"]}
            ),
            "unique_diff_id_pairs": sorted(
                {
                    (r["diff_ids"]["trigger"], r["diff_ids"]["control"])
                    for r in alignment_results
                    if r["pass"] and r["diff_ids"] is not None
                }
            ),
        },
        "warnings": warnings,
        "note": (
            "full-corpus screen; row-level reconstruction not performed. "
            "ASCII-only character filter, not English-language verification. "
            "Selection blind to model outputs. Preregistration not frozen."
        ),
    }

    (OUT_DIR / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "development_instructions.json").write_text(
        json.dumps(dev_file, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "held_out_instructions.json").write_text(
        json.dumps(held_file, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "alignment_results.json").write_text(
        json.dumps(alignment_results, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "discarded_within_dolly_components.json").write_text(
        json.dumps(discarded_components, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "stages.json").write_text(
        json.dumps(stages, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"overall_verdict": "PASS", "stages": stages, "out_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
