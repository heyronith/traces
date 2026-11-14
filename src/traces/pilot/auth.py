"""Authorization gates: scientific pilot vs calibration are independent."""

from __future__ import annotations

from typing import Any

PILOT_AUTHORIZATION_TOKEN = "STAGE1_AUTHORIZED_BY_INDEPENDENT_REVIEW"
CALIBRATION_AUTHORIZATION_TOKEN = "STAGE1_CALIBRATION_AUTHORIZED_BY_INDEPENDENT_REVIEW"


def assert_real_pilot_authorized(cfg: dict[str, Any]) -> None:
    """Require pilot.enabled + pilot token. Calibration auth never unlocks the pilot."""
    cal = cfg.get("calibration") or {}
    cal_token = cal.get("authorization_token")
    pilot_token = cfg.get("authorization_token")

    if pilot_token == CALIBRATION_AUTHORIZATION_TOKEN:
        raise RuntimeError(
            "Calibration authorization_token cannot unlock the scientific pilot."
        )
    if cal_token == PILOT_AUTHORIZATION_TOKEN and cfg.get("enabled") is not True:
        # Misplaced token in calibration block must not unlock pilot
        raise RuntimeError(
            "Pilot remains disabled; token in calibration block does not authorize Stage 1."
        )

    if cfg.get("enabled", False) is not True:
        raise RuntimeError(
            "Scientific pilot disabled (enabled=false). "
            "Real pilot entrypoint refuses execution."
        )
    if pilot_token != PILOT_AUTHORIZATION_TOKEN:
        raise RuntimeError(
            "Pilot enabled=true but authorization_token missing/invalid. "
            "Stage 1 remains gated by independent review. "
            "Calibration authorization does not substitute."
        )


def assert_pilot_refuses_while_disabled(cfg: dict[str, Any]) -> None:
    """Convenience for disabled-config checks (does not require enabled=true path)."""
    if cfg.get("enabled", False) is True:
        assert_real_pilot_authorized(cfg)
        return
    # Disabled: always refuse real execution
    raise RuntimeError(
        "Scientific pilot disabled (enabled=false). "
        "Real pilot entrypoint refuses execution."
    )


def pilot_launch_decision(cfg: dict[str, Any]) -> dict[str, Any]:
    """Local/Modal gate decision for the three-seed pilot (never uses calibration auth)."""
    base = {
        "mode": "stage1_scientific_pilot",
        "pilot_enabled": bool(cfg.get("enabled", False)),
        "calibration_enabled": bool((cfg.get("calibration") or {}).get("enabled", False)),
        "will_launch_calibration": False,
        "timeout_note": "Bounded Modal pilot function; pre-launch budget check required",
    }
    try:
        assert_real_pilot_authorized(cfg)
    except RuntimeError as e:
        return {
            **base,
            "status": "REFUSED_NOT_AUTHORIZED",
            "invoke_gpu": False,
            "reason": str(e),
            "note": "No Modal GPU pilot function invoked.",
        }
    return {
        **base,
        "status": "AUTHORIZED_TO_LAUNCH_SCIENTIFIC_PILOT",
        "invoke_gpu": True,
        "reason": "enabled=true and STAGE1_AUTHORIZED_BY_INDEPENDENT_REVIEW present",
        "note": "Calibration authorization was not used.",
    }
