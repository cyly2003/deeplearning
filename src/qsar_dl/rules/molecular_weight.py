"""Molecular-weight passive uptake limitation rule."""

from __future__ import annotations

from typing import Any, Mapping

from .base import RuleOutput, clip, get_float, join_missing


MW_ALIASES = ("MW", "mw", "molecular_weight", "molecular_weight_g_mol")


class MolecularWeightRule:
    """Provide AD/uncertainty signal for large molecules with limited passive uptake."""

    name = "molecular_weight"
    required_inputs = ["MW"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        features = {
            "rule_mw_passive_penalty": None,
            "rule_passive_uptake_factor": None,
        }
        flags: dict[str, bool | str | None] = {
            "rule_molecular_weight_applicable": False,
            "rule_molecular_weight_missing_inputs": "",
            "rule_large_molecule_flag": None,
        }

        if not config.get("enabled", True):
            flags["rule_molecular_weight_disabled"] = True
            return RuleOutput(features, {}, flags, "Molecular-weight passive uptake rule is disabled by configuration.")

        _, mw = get_float(row, MW_ALIASES)
        if mw is None:
            flags["rule_molecular_weight_missing_inputs"] = join_missing(["MW"])
            return RuleOutput(features, {}, flags, "Missing molecular weight prevents passive-uptake penalty calculation.")
        if mw <= 0:
            return RuleOutput(features, {}, flags, "Molecular-weight rule is not applicable to non-positive MW.")

        threshold = float(config.get("passive_threshold_mw", 600.0))
        width = float(config.get("passive_penalty_width", 400.0))
        penalty = clip((mw - threshold) / width, 0.0, 1.0) if width > 0 else 0.0

        features.update(
            {
                "rule_mw_passive_penalty": penalty,
                "rule_passive_uptake_factor": 1.0 - penalty,
            }
        )
        flags.update(
            {
                "rule_molecular_weight_applicable": True,
                "rule_large_molecule_flag": bool(penalty > 0.0),
            }
        )
        return RuleOutput(
            features,
            {},
            flags,
            "Computed passive uptake limitation signal from molecular weight; this is an AD/uncertainty feature only.",
        )
