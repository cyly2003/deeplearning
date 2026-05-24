"""Exposure-duration rule for acute aquatic endpoints."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .base import RuleOutput, get_float, get_text, join_missing


DURATION_ALIASES = ("duration_h", "duration_hour", "duration_hours", "exposure_duration_h")
PTOX_ALIASES = ("y_obs", "target_ptox", "ptox", "pTox", "y_pred", "predicted_ptox")
ENDPOINT_ALIASES = ("endpoint", "endpoint_family", "effect_endpoint", "test_endpoint")
TAXON_ALIASES = ("eco_group", "taxon", "taxonomy_class", "species_group", "organism_group")
STANDARD_KEY_ALIASES = ("duration_standard_key", "standard_duration_key")


DEFAULT_STANDARD_HOURS = {
    "fish_lc50": 96.0,
    "daphnia_ec50": 48.0,
    "daphnia_lc50": 48.0,
    "algae_ec50": 72.0,
}


def infer_standard_duration_key(row: Mapping[str, Any]) -> str | None:
    """Infer a standard acute duration key from endpoint and broad taxon labels."""

    _, explicit = get_text(row, STANDARD_KEY_ALIASES)
    if explicit:
        return explicit.lower()

    _, endpoint = get_text(row, ENDPOINT_ALIASES)
    endpoint_text = (endpoint or "").lower()
    if "loec" in endpoint_text or "noec" in endpoint_text:
        return None

    _, taxon = get_text(row, TAXON_ALIASES)
    taxon_text = (taxon or "").lower()
    if "fish" in taxon_text and "lc" in endpoint_text:
        return "fish_lc50"
    if any(token in taxon_text for token in ("daphnia", "cladocera", "crustacea", "invertebrate")):
        if "ec" in endpoint_text:
            return "daphnia_ec50"
        if "lc" in endpoint_text:
            return "daphnia_lc50"
    if any(token in taxon_text for token in ("algae", "alga", "cyanobacteria", "cyanophyta")) and "ec" in endpoint_text:
        return "algae_ec50"
    return None


class DurationRule:
    """Flag short exposure duration and provide a non-final pTox adjustment candidate."""

    name = "duration"
    required_inputs = ["duration_h"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        features = {
            "rule_duration_ratio": None,
            "rule_short_duration_flag": None,
        }
        corrections = {"rule_duration_ptox_adjustment_candidate": None}
        flags: dict[str, bool | str | None] = {
            "rule_duration_applicable": False,
            "rule_duration_missing_inputs": "",
        }

        if not config.get("enabled", True):
            flags["rule_duration_disabled"] = True
            return RuleOutput(features, corrections, flags, "Duration rule is disabled by configuration.")

        _, endpoint = get_text(row, ENDPOINT_ALIASES)
        endpoint_text = (endpoint or "").lower()
        if "loec" in endpoint_text or "noec" in endpoint_text:
            return RuleOutput(
                features,
                corrections,
                flags,
                "LOEC/NOEC endpoints are not hard-corrected by the acute exposure-duration rule.",
            )

        _, duration_h = get_float(row, DURATION_ALIASES)
        if duration_h is None:
            flags["rule_duration_missing_inputs"] = join_missing(["duration_h"])
            return RuleOutput(features, corrections, flags, "Missing exposure duration prevents duration-ratio calculation.")
        if duration_h <= 0:
            return RuleOutput(features, corrections, flags, "Duration rule is not applicable to non-positive exposure durations.")

        key = infer_standard_duration_key(row)
        standard_hours = {**DEFAULT_STANDARD_HOURS, **config.get("standard_hours", {})}
        d_std = float(standard_hours[key]) if key in standard_hours else None
        if d_std is None:
            flags["rule_duration_missing_inputs"] = join_missing(["endpoint_or_taxon"])
            return RuleOutput(
                features,
                corrections,
                flags,
                "Cannot infer a standard acute duration without a supported endpoint/taxon combination.",
            )

        ratio = duration_h / d_std
        gamma = config.get("gamma")
        if gamma is None:
            gamma_grid = config.get("gamma_grid", [0.25])
            gamma = gamma_grid[0] if gamma_grid else 0.25
        gamma = max(0.0, float(gamma))

        _, ptox = get_float(row, PTOX_ALIASES)
        adjustment = gamma * math.log10(d_std / duration_h) if duration_h < d_std else 0.0

        features.update(
            {
                "rule_duration_ratio": ratio,
                "rule_short_duration_flag": int(ratio < 1.0),
            }
        )
        corrections["rule_duration_ptox_adjustment_candidate"] = adjustment if ptox is not None else adjustment
        flags["rule_duration_applicable"] = True
        return RuleOutput(
            features,
            corrections,
            flags,
            "Computed acute exposure-duration ratio; short-duration candidate adjustment is non-negative by construction.",
        )
