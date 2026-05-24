"""Water solubility and saturation-limit rule."""

from __future__ import annotations

from typing import Any, Mapping

from .base import RuleOutput, get_float, join_missing


PTOX_ALIASES = ("y_pred", "predicted_ptox", "target_ptox", "ptox", "pTox", "y_obs")
MW_ALIASES = ("MW", "mw", "molecular_weight", "molecular_weight_g_mol")
SOLUBILITY_ALIASES = (
    "water_solubility_mg_l",
    "water_solubility_mg/L",
    "solubility_mg_l",
    "S_w_mg_l",
    "sw_mg_l",
)


class SolubilityRule:
    """Flag predictions whose implied effect concentration exceeds water solubility."""

    name = "solubility"
    required_inputs = ["y_pred", "MW", "water_solubility_mg_l"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        features = {
            "rule_solubility_ratio": None,
            "rule_near_saturation_flag": None,
            "rule_no_effects_at_saturation_flag": None,
        }
        flags: dict[str, bool | str | None] = {
            "rule_solubility_applicable": False,
            "rule_solubility_missing_inputs": "",
        }

        if not config.get("enabled", True):
            flags["rule_solubility_disabled"] = True
            return RuleOutput(features, {}, flags, "Solubility saturation rule is disabled by configuration.")

        _, ptox = get_float(row, PTOX_ALIASES)
        _, mw = get_float(row, MW_ALIASES)
        _, solubility = get_float(row, SOLUBILITY_ALIASES)

        missing = []
        if ptox is None:
            missing.append("y_pred")
        if mw is None:
            missing.append("MW")
        if solubility is None:
            missing.append("water_solubility_mg_l")
        if missing:
            flags["rule_solubility_missing_inputs"] = join_missing(missing)
            return RuleOutput(
                features,
                {},
                flags,
                "Missing pTox, molecular weight, or water solubility prevents saturation-limit calculation.",
            )

        if mw <= 0 or solubility <= 0:
            return RuleOutput(
                features,
                {},
                flags,
                "Solubility rule is not applicable because molecular weight or water solubility is non-positive.",
            )

        effect_mol_l = 10 ** (-ptox)
        effect_mg_l = effect_mol_l * mw * 1000.0
        ratio = effect_mg_l / solubility
        no_effects_ratio = float(config.get("no_effects_ratio", 10.0))
        near_ratio = float(config.get("near_saturation_ratio", 1.0))

        features.update(
            {
                "rule_solubility_ratio": ratio,
                "rule_near_saturation_flag": int(near_ratio <= ratio < no_effects_ratio),
                "rule_no_effects_at_saturation_flag": int(ratio >= no_effects_ratio),
            }
        )
        flags["rule_solubility_applicable"] = True
        return RuleOutput(
            features,
            {},
            flags,
            "Compared predicted effect concentration in mg/L against water solubility to flag saturation limits.",
        )
