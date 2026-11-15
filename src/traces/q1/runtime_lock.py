"""Built-image runtime lock: generate and validate wheel-hashed freezes.

Production lock must be generated from the **actual intended scientific image**.
This module supports synthetic local fixtures for procedure verification only.
Do **not** claim a production image lock exists until generated from that image.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from traces.q1 import FROZEN_PROTOCOL_VERSION, FROZEN_SCIENTIFIC_COMMIT
from traces.q1.runtime_spec import DIRECT_PINS_FILE, DIRECT_PINS_SHA256, verify_direct_pins_file

LOCK_SCHEMA_VERSION = "1.0"


@dataclass
class WheelRecord:
    name: str
    version: str
    filename: str
    sha256: str


@dataclass
class RuntimeLockDocument:
    schema_version: str = LOCK_SCHEMA_VERSION
    protocol_version: str = FROZEN_PROTOCOL_VERSION
    frozen_scientific_commit: str = FROZEN_SCIENTIFIC_COMMIT
    created_utc: str = ""
    python: dict[str, Any] = field(default_factory=dict)
    platform: dict[str, Any] = field(default_factory=dict)
    cuda: dict[str, Any] = field(default_factory=dict)
    direct_pins: dict[str, Any] = field(default_factory=dict)
    wheels: list[dict[str, Any]] = field(default_factory=list)
    source: str = "synthetic_fixture"  # or "built_image"
    production_image_lock: bool = False
    intended_image: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_python_runtime_info() -> dict[str, Any]:
    return {
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def collect_platform_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def collect_cuda_info() -> dict[str, Any]:
    """Best-effort CUDA metadata; never initializes GPU work."""
    info: dict[str, Any] = {
        "torch_cuda_available": False,
        "torch_version": None,
        "cuda_version": None,
        "device_count": 0,
        "note": "CPU-safe probe only",
    }
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if info["torch_cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
    except Exception as e:  # noqa: BLE001 — lock prep must not crash
        info["error"] = str(e)
    return info


def parse_pip_freeze_lines(text: str) -> list[tuple[str, str]]:
    """Parse ``name==version`` lines; ignore editables and comments."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-e "):
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            out.append((name.strip(), ver.strip()))
    return out


def generate_lock_from_wheel_dir(
    wheel_dir: Path | str,
    *,
    repo_root: Path | str,
    pip_freeze_text: str | None = None,
    source: str = "synthetic_fixture",
    production_image_lock: bool = False,
    extra_notes: Sequence[str] | None = None,
) -> RuntimeLockDocument:
    """Generate a lock document from local wheel files + optional freeze text.

    For production: run inside the built image with ``source='built_image'`` and
    ``production_image_lock=True`` only after verifying the intended image tag.
    """
    wdir = Path(wheel_dir)
    if not wdir.is_dir():
        raise FileNotFoundError(f"wheel_dir not found: {wdir}")

    pins = verify_direct_pins_file(repo_root)
    wheels: list[dict[str, Any]] = []
    for path in sorted(wdir.glob("*.whl")):
        # filename convention: name-version-...whl (PEP 427)
        digest = _sha256_file(path)
        stem = path.name[:-4]
        parts = stem.split("-")
        name = parts[0] if parts else path.name
        version = parts[1] if len(parts) > 1 else ""
        wheels.append(
            {
                "name": name,
                "version": version,
                "filename": path.name,
                "sha256": digest,
            }
        )

    freeze_pkgs = parse_pip_freeze_lines(pip_freeze_text or "")
    notes = list(extra_notes or [])
    if not production_image_lock:
        notes.append(
            "NOT a production image lock — synthetic/local fixture only"
        )
    if not pins.get("match"):
        notes.append("direct pins SHA mismatch")

    return RuntimeLockDocument(
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        python=collect_python_runtime_info(),
        platform=collect_platform_info(),
        cuda=collect_cuda_info(),
        direct_pins=pins,
        wheels=wheels,
        source=source,
        production_image_lock=production_image_lock,
        notes=notes
        + [
            f"pip_freeze_packages_parsed={len(freeze_pkgs)}",
            f"direct_pins_file={DIRECT_PINS_FILE}",
            f"direct_pins_expected_sha256={DIRECT_PINS_SHA256}",
        ],
    )


def write_lock_document(doc: RuntimeLockDocument, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc.to_dict(), indent=2) + "\n")
    return p


def load_lock_document(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def validate_lock_document(
    doc: Mapping[str, Any],
    *,
    wheel_dir: Path | str | None = None,
    require_production: bool = False,
    require_direct_pins_match: bool = True,
) -> dict[str, Any]:
    """Validate a lock document; optionally re-hash wheels on disk."""
    failures: list[str] = []
    if doc.get("schema_version") != LOCK_SCHEMA_VERSION:
        failures.append(
            f"schema_version {doc.get('schema_version')!r} != {LOCK_SCHEMA_VERSION!r}"
        )
    if require_production and not doc.get("production_image_lock"):
        failures.append("production_image_lock is False; refusing as production lock")
    if require_direct_pins_match:
        dp = doc.get("direct_pins") or {}
        if not dp.get("match"):
            failures.append("direct_pins.match is False")
        if dp.get("expected_sha256") != DIRECT_PINS_SHA256:
            failures.append("direct_pins expected SHA does not match module constant")

    wheels = doc.get("wheels") or []
    if not wheels:
        failures.append("no wheels recorded")

    if wheel_dir is not None:
        wdir = Path(wheel_dir)
        by_name = {w["filename"]: w for w in wheels if "filename" in w}
        for fname, rec in by_name.items():
            path = wdir / fname
            if not path.is_file():
                failures.append(f"missing wheel file: {fname}")
                continue
            digest = _sha256_file(path)
            if digest != rec.get("sha256"):
                failures.append(
                    f"wheel SHA mismatch {fname}: {digest} != {rec.get('sha256')}"
                )
        # Extra files on disk not in lock are a soft warning
        extras = [p.name for p in wdir.glob("*.whl") if p.name not in by_name]
    else:
        extras = []

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "n_wheels": len(wheels),
        "extra_wheels_on_disk": extras,
        "production_image_lock": bool(doc.get("production_image_lock")),
        "source": doc.get("source"),
    }


# Direct-pin packages that a production lock must cover (name prefixes / keys)
REQUIRED_PRODUCTION_PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "peft",
)


def claim_production_lock_exists(
    doc: Mapping[str, Any] | None,
    *,
    wheel_dir: Path | str | None = None,
    require_rehash: bool = True,
    intended_image: Mapping[str, Any] | None = None,
) -> bool:
    """True only after full validation — flags alone are insufficient.

    Rejects forged ``production_image_lock=True`` / ``source=built_image`` without
    complete wheel coverage, SHA checks, direct-pin linkage, and image metadata.
    """
    if not doc:
        return False
    if not (bool(doc.get("production_image_lock")) and doc.get("source") == "built_image"):
        return False

    # Intended-image metadata must be present and non-empty
    meta = doc.get("intended_image") or intended_image
    if not isinstance(meta, Mapping):
        return False
    for key in ("image_tag", "image_digest", "python_version", "cuda_version"):
        if not meta.get(key):
            return False

    # Direct-pin linkage
    dp = doc.get("direct_pins") or {}
    if not dp.get("match"):
        return False
    if dp.get("expected_sha256") != DIRECT_PINS_SHA256:
        return False

    wheels = doc.get("wheels") or []
    if len(wheels) < len(REQUIRED_PRODUCTION_PACKAGES):
        return False
    names = {str(w.get("name", "")).lower().replace("-", "_") for w in wheels}
    for req in REQUIRED_PRODUCTION_PACKAGES:
        if not any(req in n for n in names):
            return False
    for w in wheels:
        if not w.get("sha256") or not w.get("filename"):
            return False

    # Structural validation (+ optional on-disk rehash)
    val = validate_lock_document(
        doc,
        wheel_dir=wheel_dir if require_rehash else None,
        require_production=True,
        require_direct_pins_match=True,
    )
    if not val["ok"]:
        return False

    # Explicit forge guard: schema + protocol linkage
    if doc.get("schema_version") != LOCK_SCHEMA_VERSION:
        return False
    if doc.get("frozen_scientific_commit") != FROZEN_SCIENTIFIC_COMMIT:
        return False
    return True
