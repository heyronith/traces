#!/usr/bin/env python3
"""Re-verify frozen pilot dataset hashes (does not regenerate unless --write).

The authoritative freeze already lives in data/pilot/. Regeneration is gated to
avoid accidental hash changes after pre-registration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traces.pilot.dataset import dataset_summary, verify_manifest_hashes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--write",
        action="store_true",
        help="Forbidden in normal workflow; freeze is already committed.",
    )
    args = p.parse_args()
    if args.write:
        raise SystemExit(
            "Refusing to regenerate the pre-registered dataset from this script. "
            "Edit data/pilot/ only under a new versioned manifest name."
        )
    checks = verify_manifest_hashes()
    summary = dataset_summary()
    print(json.dumps({"hash_checks": checks, "summary": summary}, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
