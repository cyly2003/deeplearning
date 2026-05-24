"""Weak-acid/weak-base ionization rule."""

from __future__ import annotations

from typing import Any, Mapping

from .base import RuleOutput, get_float, get_text, join_missing


PH_ALIASES = ("pH", "ph", "test_ph")
PKA_ACID_ALIASES = ("pKa_acid", "pka_acid", "acid_pka")
PKA_BASE_ALIASES = ("pKa_base", "pka_base", "base_pka")
ION_TYPE_ALIASES = ("ionization_type", "acid_base_type", "compound_acid_base_class")
LOGD_ALIASES = ("logD", "logd", "LogD", "logD_pH", "logd_ph")


class IonizationRule:
    """Estimate neutral fraction for weak acids and bases at test pH."""

    name = "ionization"
    required_inputs = ["pH", "pKa_acid_or_pKa_base"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        features = {"rule_neutral_fraction": None}
        flags: dict[str, bool | str | None] = {
            "rule_ionization_applicable": False,
            "rule_ionization_missing_inputs": "",
            "rule_ionization_flag": None,
            "rule_logkow_replaced_by_logd_flag": None,
        }

        if not config.get("enabled", True):
            flags["rule_ionization_disabled"] = True
            return RuleOutput(features, {}, flags, "Ionization rule is disabled by configuration.")

        _, ion_type = get_text(row, ION_TYPE_ALIASES)
        ion_type_text = (ion_type or "").lower()
        if ion_type_text in {"neutral", "nonionizable", "non-ionizable", "none"}:
            features["rule_neutral_fraction"] = 1.0
            flags.update(
                {
                    "rule_ionization_applicable": False,
                    "rule_ionization_flag": False,
                    "rule_logkow_replaced_by_logd_flag": False,
                }
            )
            return RuleOutput(
                features,
                {},
                flags,
                "Compound is marked non-ionizable/neutral, so no weak-acid/base fraction calculation is applied.",
            )

        _, ph = get_float(row, PH_ALIASES)
        _, pka_acid = get_float(row, PKA_ACID_ALIASES)
        _, pka_base = get_float(row, PKA_BASE_ALIASES)

        missing = []
        if ph is None:
            missing.append("pH")
        if pka_acid is None and pka_base is None:
            missing.append("pKa_acid_or_pKa_base")
        if missing:
            flags["rule_ionization_missing_inputs"] = join_missing(missing)
            return RuleOutput(features, {}, flags, "Missing pH or pKa prevents neutral-fraction calculation.")

        if "base" in ion_type_text and pka_base is not None:
            neutral_fraction = 1.0 / (1.0 + 10 ** (pka_base - ph))
            acid_base_used = "weak base"
        elif pka_acid is not None:
            neutral_fraction = 1.0 / (1.0 + 10 ** (ph - pka_acid))
            acid_base_used = "weak acid"
        elif pka_base is not None:
            neutral_fraction = 1.0 / (1.0 + 10 ** (pka_base - ph))
            acid_base_used = "weak base"
        else:
            flags["rule_ionization_missing_inputs"] = join_missing(["pKa_acid_or_pKa_base"])
            return RuleOutput(features, {}, flags, "Missing pKa prevents neutral-fraction calculation.")

        _, logd = get_float(row, LOGD_ALIASES)
        threshold = float(config.get("ionized_fraction_threshold", 0.5))
        features["rule_neutral_fraction"] = neutral_fraction
        flags.update(
            {
                "rule_ionization_applicable": True,
                "rule_ionization_flag": bool(neutral_fraction < threshold),
                "rule_logkow_replaced_by_logd_flag": bool(logd is not None),
            }
        )
        return RuleOutput(
            features,
            {},
            flags,
            f"Computed neutral fraction for a {acid_base_used}; low neutral fraction flags ionization-sensitive hydrophobicity.",
        )
