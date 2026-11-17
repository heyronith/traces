"""Built-image runtime lock: generate and validate wheel-hashed freezes.

Production lock must be generated from the **actual intended scientific image**.
This module supports synthetic local fixtures for procedure verification only.
Do **not** claim a production image lock exists until generated from that image.
Synthetic fixture validation must never substitute for production verification.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence  # Mapping used by installed-runtime verify

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from traces.q1 import FROZEN_PROTOCOL_VERSION, FROZEN_SCIENTIFIC_COMMIT
from traces.q1.frozen_artifacts import parse_direct_pin_requirements
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
    # Exact inventory of *installed* packages from the intended image's pip
    # freeze, not merely downloadable wheels or direct requirements.
    installed_distributions: list[dict[str, str]] = field(default_factory=list)
    pip_freeze_sha256: str = ""
    pip_freeze_text: str = ""
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
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return info


def parse_pip_freeze_lines(text: str) -> list[tuple[str, str]]:
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
    intended_image: Mapping[str, Any] | None = None,
    extra_notes: Sequence[str] | None = None,
) -> RuntimeLockDocument:
    wdir = Path(wheel_dir)
    if not wdir.is_dir():
        raise FileNotFoundError(f"wheel_dir not found: {wdir}")

    pins = verify_direct_pins_file(repo_root)
    wheels: list[dict[str, Any]] = []
    for path in sorted(wdir.glob("*.whl")):
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
    if not production_image_lock or source != "built_image":
        notes.append("NOT a production image lock — synthetic/local fixture only")
    if not pins.get("match"):
        notes.append("direct pins SHA mismatch")

    return RuntimeLockDocument(
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        python=collect_python_runtime_info(),
        platform=collect_platform_info(),
        cuda=collect_cuda_info(),
        direct_pins=pins,
        wheels=wheels,
        installed_distributions=[
            {"name": name, "version": version} for name, version in freeze_pkgs
        ],
        pip_freeze_sha256=(
            hashlib.sha256(pip_freeze_text.encode("utf-8")).hexdigest()
            if pip_freeze_text is not None else ""
        ),
        pip_freeze_text=pip_freeze_text or "",
        source=source,
        production_image_lock=production_image_lock,
        intended_image=dict(intended_image or {}),
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
    if require_production and not _complete_installed_wheel_inventory(doc):
        failures.append(
            "production lock missing complete installed-distribution / "
            "hashed-wheel coverage or pip-freeze binding"
        )

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
        extras = [p.name for p in wdir.glob("*.whl") if p.name not in by_name]
    else:
        extras = []
    if require_production and extras:
        failures.append(f"unexpected extra wheels in production evidence: {extras}")

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "n_wheels": len(wheels),
        "extra_wheels_on_disk": extras,
        "production_image_lock": bool(doc.get("production_image_lock")),
        "source": doc.get("source"),
    }


def _normalize_pkg_name(name: str) -> str:
    return canonicalize_name(name)


def _matches_frozen_pin(installed: str, pinned: str) -> bool:
    """PEP 440 == public-version matching permits a recorded +local build.

    This helper only answers the public pin question. CUDA builds are bound
    separately to exact torch local suffix, wheel hash and intended GPU image.
    """
    try:
        return Version(installed) in SpecifierSet(f"=={pinned}")
    except (InvalidVersion, ValueError, TypeError):
        return False


def _complete_installed_wheel_inventory(doc: Mapping[str, Any]) -> bool:
    """Fail closed unless every installed distribution has its exact wheel.

    A download pool of a subset of installed wheels is not a transitive lock.
    Entries are required from a pip freeze captured *inside* the built image.
    """
    installed = doc.get("installed_distributions")
    wheels = doc.get("wheels")
    freeze_sha = doc.get("pip_freeze_sha256")
    freeze_text = doc.get("pip_freeze_text")
    if not isinstance(freeze_text, str) or not freeze_text:
        return False
    if hashlib.sha256(freeze_text.encode("utf-8")).hexdigest() != freeze_sha:
        return False
    parsed_freeze = parse_pip_freeze_lines(freeze_text)
    active_lines = [
        line for line in freeze_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(parsed_freeze) != len(active_lines):
        return False
    if (not isinstance(installed, list) or not installed
            or not isinstance(wheels, list) or not wheels):
        return False
    if (not isinstance(freeze_sha, str) or len(freeze_sha) != 64
            or not all(c in "0123456789abcdef" for c in freeze_sha.lower())):
        return False

    by_installed: dict[str, str] = {}
    for rec in installed:
        if not isinstance(rec, Mapping):
            return False
        name, version = rec.get("name"), rec.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str):
            return False
        try:
            norm = _normalize_pkg_name(name)
            Version(version)
        except (ValueError, InvalidVersion, TypeError):
            return False
        if norm in by_installed:
            return False
        by_installed[norm] = version

    by_wheel: dict[str, str] = {}
    for rec in wheels:
        if not isinstance(rec, Mapping):
            return False
        name, version, filename = (
            rec.get("name"), rec.get("version"), rec.get("filename")
        )
        if not all(isinstance(v, str) and v for v in (name, version, filename)):
            return False
        try:
            parsed_name, parsed_version, _build, _tags = parse_wheel_filename(filename)
            norm = _normalize_pkg_name(name)
            ver = Version(version)
            if norm != parsed_name or ver != parsed_version:
                return False
        except (ValueError, InvalidVersion, TypeError):
            return False
        if norm in by_wheel:
            return False
        by_wheel[norm] = version

    parsed_installed: dict[str, str] = {}
    for name, version in parsed_freeze:
        norm = _normalize_pkg_name(name)
        if norm in parsed_installed:
            return False
        parsed_installed[norm] = version
    if parsed_installed != by_installed:
        return False
    if set(by_installed) != set(by_wheel):
        return False
    return all(
        Version(by_installed[name]) == Version(by_wheel[name])
        for name in by_installed
    )


def _torch_cuda_build_matches_image(doc: Mapping[str, Any]) -> bool:
    """Lock CUDA-tagged torch bytes to the actual intended image/probe."""
    meta = doc.get("intended_image") or {}
    cuda = doc.get("cuda") or {}
    if not isinstance(meta, Mapping) or not isinstance(cuda, Mapping):
        return False
    cv = str(meta.get("cuda_version", ""))
    if not cv or str(cuda.get("cuda_version")) != cv:
        return False
    try:
        major, minor = cv.split(".")
        if not major.isdigit() or not minor.isdigit():
            return False
        expected_local = f"cu{int(major)}{int(minor)}"
        versions = {
            w.get("version") for w in doc["wheels"]
            if _normalize_pkg_name(str(w.get("name", ""))) == "torch"
        }
        if len(versions) != 1:
            return False
        torch_version = next(iter(versions))
        if Version(torch_version).local != expected_local:
            return False
        if meta.get("torch_version") != torch_version:
            return False
        if cuda.get("torch_version") != torch_version:
            return False
    except (KeyError, ValueError, InvalidVersion, TypeError):
        return False
    probe = cuda.get("a10g_probe")
    if not isinstance(probe, Mapping) or probe.get("ok") is not True:
        return False
    if (str(probe.get("cuda_version")) != cv
            or probe.get("device_name") != meta.get("gpu_device_name")):
        return False
    return True


def claim_production_lock_exists(
    doc: Mapping[str, Any] | None,
    *,
    wheel_dir: Path | str | None = None,
    require_rehash: bool = True,
    intended_image: Mapping[str, Any] | None = None,
    pins_file: Path | str | None = None,
) -> bool:
    """True only after full independent verification.

    Flags ``production_image_lock`` / ``source=built_image`` alone are never
    sufficient. Production claims **require** on-disk wheel rehashing
    (``require_rehash=True`` and a readable ``wheel_dir``). Synthetic fixtures
    cannot satisfy this claim.
    """
    if not doc:
        return False
    if doc.get("source") == "synthetic_fixture":
        return False
    if not (bool(doc.get("production_image_lock")) and doc.get("source") == "built_image"):
        return False

    # Rehash is mandatory for production — refuse unverified wheel evidence
    if not require_rehash or wheel_dir is None:
        return False
    wdir = Path(wheel_dir)
    if not wdir.is_dir():
        return False

    meta = doc.get("intended_image") or intended_image
    if not isinstance(meta, Mapping):
        return False
    for key in ("image_tag", "image_digest", "python_version", "cuda_version"):
        if not meta.get(key):
            return False
    if meta.get("python_version") != "3.11.15":
        return False

    dp = doc.get("direct_pins") or {}
    if not dp.get("match"):
        return False
    if dp.get("expected_sha256") != DIRECT_PINS_SHA256:
        return False
    if dp.get("sha256") != DIRECT_PINS_SHA256:
        return False

    # All frozen direct dependencies with exact versions
    pins_path = Path(pins_file) if pins_file else None
    if pins_path is None or not pins_path.is_file():
        return False
    required = parse_direct_pin_requirements(pins_path.read_text())
    if not required:
        return False

    wheels = doc.get("wheels") or []
    if not wheels:
        return False
    if not _complete_installed_wheel_inventory(doc):
        return False
    if not _torch_cuda_build_matches_image(doc):
        return False

    # Index wheels by normalized name; require PEP 440 public-version matches
    # for pinned deps while binding CUDA local wheel bytes separately.
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for w in wheels:
        if not w.get("sha256") or not w.get("filename") or not w.get("version"):
            return False
        # Reject obviously fabricated all-zero / empty hashes
        sha = str(w["sha256"])
        if len(sha) != 64 or sha == "0" * 64 or not all(c in "0123456789abcdef" for c in sha.lower()):
            return False
        nn = _normalize_pkg_name(str(w.get("name", "")))
        by_norm.setdefault(nn, []).append(w)

    for pkg, ver in required.items():
        # M4: exact normalized package-name match only — reject loose substring matches
        cands = by_norm.get(pkg)
        if not cands:
            return False
        if not any(_matches_frozen_pin(str(w.get("version")), ver) for w in cands):
            return False

    val = validate_lock_document(
        doc,
        wheel_dir=wdir,
        require_production=True,
        require_direct_pins_match=True,
    )
    if not val["ok"] or val["extra_wheels_on_disk"]:
        return False

    if doc.get("schema_version") != LOCK_SCHEMA_VERSION:
        return False
    if doc.get("frozen_scientific_commit") != FROZEN_SCIENTIFIC_COMMIT:
        return False
    if doc.get("protocol_version") != FROZEN_PROTOCOL_VERSION:
        return False
    return True


def verify_installed_runtime_environment(
    *,
    pins_file: Path | str,
    installed: Mapping[str, str] | None = None,
    require_python: str = "3.11.15",
    intended_image: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """M4: verify executable runtime + exact pinned direct versions.

    ``installed`` maps normalized package name → version. When omitted, attempts
    ``importlib.metadata``. Loose substring package-name matches are rejected.
    Does **not** claim production verification without an actual built image.
    """
    failures: list[str] = []
    py = platform.python_version()
    if py != require_python:
        failures.append(f"python_version {py!r} != required {require_python!r}")

    exe = Path(sys.executable).resolve()
    pins_path = Path(pins_file)
    if not pins_path.is_file():
        return {
            "ok": False,
            "failures": [f"pins file missing: {pins_path}"],
            "production_claim": False,
            "executable": str(exe),
            "python_version": py,
        }
    required = parse_direct_pin_requirements(pins_path.read_text())

    if installed is None:
        installed = {}
        try:
            from importlib import metadata as importlib_metadata

            for pkg in required:
                try:
                    installed[pkg] = importlib_metadata.version(pkg)
                except importlib_metadata.PackageNotFoundError:
                    pass
        except Exception as e:  # noqa: BLE001
            failures.append(f"could not inspect installed packages: {e}")

    # Exact name match only
    norm_installed = {_normalize_pkg_name(k): str(v) for k, v in (installed or {}).items()}
    for pkg, ver in required.items():
        got = norm_installed.get(pkg)
        if got is None:
            failures.append(f"required package {pkg!r} not installed")
        elif not _matches_frozen_pin(got, ver):
            failures.append(f"{pkg} version {got!r} != pinned {ver!r}")

    image_ok = False
    image_notes: list[str] = []
    if intended_image is None:
        image_notes.append(
            "intended_image identity not supplied — refuse production claim"
        )
    else:
        for key in ("image_tag", "image_digest", "python_version", "cuda_version"):
            if not intended_image.get(key):
                failures.append(f"intended_image missing {key}")
        if intended_image.get("python_version") and intended_image.get(
            "python_version"
        ) != require_python:
            failures.append("intended_image.python_version mismatch")
        image_ok = (
            bool(intended_image.get("image_tag"))
            and bool(intended_image.get("image_digest"))
            and intended_image.get("python_version") == require_python
        )

    return {
        "ok": len(failures) == 0 and image_ok,
        "failures": failures,
        "production_claim": False,  # never claim without built-image lock + rehash
        "executable": str(exe),
        "python_version": py,
        "direct_pins_exact": len(failures) == 0,
        "intended_image_present": image_ok,
        "notes": image_notes
        + [
            "Production verification requires actual built image + wheel rehash; "
            "this check alone is insufficient"
        ],
    }
