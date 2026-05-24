"""Water-phase hydrophobicity baseline toxicity rule."""

from __future__ import annotations

from typing import Any, Mapping

from .base import RuleOutput, as_str, clip, get_float, get_text, join_missing


LOGKOW_ALIASES = ("logKow", "logkow", "LogKow", "logKOW", "logP", "LogP", "xlogp")
ENDPOINT_ALIASES = ("endpoint", "endpoint_family", "effect_endpoint", "test_endpoint")
MOA_ALIASES = ("moa", "MoA", "mode_of_action", "chemical_class")


class AquaticHydrophobicityRule:
    """Baseline/narcosis pTox increases with effective logKow."""

    name = "aquatic_hydrophobicity"
    required_inputs = ["logKow"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        features = {"rule_aq_logkow_baseline_ptox": None}
        flags: dict[str, bool | str | None] = {
            "rule_aquatic_hydrophobicity_applicable": False,
            "rule_aquatic_hydrophobicity_missing_inputs": "",
            "rule_aq_logkow_applicable": False,
            "rule_aq_logkow_missing_inputs": "",
            "rule_aq_logkow_slope_positive": None,
        }

        if not config.get("enabled", True):
            flags["rule_aquatic_hydrophobicity_disabled"] = True
            return RuleOutput(
                features,
                {},
                flags,
                "Aquatic hydrophobicity baseline rule is disabled by configuration.",
            )

        _, logkow = get_float(row, LOGKOW_ALIASES)
        if logkow is None:
            missing = join_missing(["logKow"])
            flags["rule_aquatic_hydrophobicity_missing_inputs"] = missing
            flags["rule_aq_logkow_missing_inputs"] = missing
            return RuleOutput(
                features,
                {},
                flags,
                "Missing logKow/logP prevents baseline narcosis pTox calculation.",
            )

        _, endpoint = get_text(row, ENDPOINT_ALIASES)
        endpoint_text = (endpoint or "").upper()
        if "LOEC" in endpoint_text or "NOEC" in endpoint_text:
            return RuleOutput(
                features,
                {},
                flags,
                "Baseline acute hydrophobicity rule is not applied to chronic threshold endpoints.",
            )

        _, moa = get_text(row, MOA_ALIASES)
        moa_text = (moa or "").lower()
        if moa_text and any(token in moa_text for token in ("metal", "inorganic", "reactive", "specific")):
            return RuleOutput(
                features,
                {},
                flags,
                "Baseline narcosis rule is not applied to reactive, inorganic, metal, or specific-acting classes.",
            )

        slope = float(config.get("b_group", config.get("slope", 0.45)))
        intercept = float(config.get("a_group", config.get("intercept", 2.0)))
        if slope <= 0:
            raise ValueError("aquatic_hydrophobicity b_group/slope must be positive.")

        logkow_min = float(config.get("logkow_min", -1.0))
        logkow_max = float(config.get("logkow_max", 6.0))
        baseline = intercept + slope * clip(logkow, logkow_min, logkow_max)

        features["rule_aq_logkow_baseline_ptox"] = baseline
        flags.update(
            {
                "rule_aquatic_hydrophobicity_applicable": True,
                "rule_aq_logkow_applicable": True,
                "rule_aq_logkow_slope_positive": True,
            }
        )
        return RuleOutput(
            features,
            {},
            flags,
            "Computed baseline/narcosis pTox from clipped logKow with a positive hydrophobicity slope.",
        )
