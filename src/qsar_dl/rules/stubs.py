"""Clear placeholders for rule-layer methods not yet calibrated."""

from __future__ import annotations

from typing import Any, Mapping

from .base import RuleOutput


def _stub_output(name: str, features: dict[str, float | int | None], explanation: str) -> RuleOutput:
    return RuleOutput(
        features=features,
        corrections={},
        flags={
            f"rule_{name}_applicable": False,
            f"rule_{name}_missing_inputs": "not_implemented",
            f"rule_{name}_disabled": True,
        },
        explanation=explanation,
    )


class ChemicalActivityRule:
    name = "chemical_activity"
    required_inputs = ["C_free_mol_l", "S_w_mol_l"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        return _stub_output(
            self.name,
            {
                "rule_chemical_activity": None,
                "rule_activity_low_penalty": None,
                "rule_activity_high_flag": None,
            },
            "Chemical-activity rule is a TODO placeholder until free concentration and solubility in mol/L are standardized.",
        )


class MoaExcessToxicityRule:
    name = "moa_excess_toxicity"
    required_inputs = ["y_obs_or_pred", "rule_aq_logkow_baseline_ptox"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        return _stub_output(
            self.name,
            {
                "rule_toxic_ratio": None,
                "rule_excess_toxicity_flag": None,
                "rule_moa_positive_residual": None,
            },
            "MoA excess-toxicity rule is a TODO placeholder until MoA/ToxCast/alert inputs are stabilized.",
        )


class TktdRule:
    name = "tktd"
    required_inputs = ["duration_h", "k_e_or_k_raw"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        return _stub_output(
            self.name,
            {
                "rule_tktd_fss_obs": None,
                "rule_tktd_fss_std": None,
                "rule_tktd_duration_adjustment_candidate": None,
            },
            "TKTD accumulation rule is a TODO placeholder; duration and TKTD corrections must not be double-counted.",
        )


class VolatilityRule:
    name = "volatility"
    required_inputs = ["duration_h", "Henry_constant_or_logH", "vapor_pressure"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        return _stub_output(
            self.name,
            {
                "rule_volatility_loss_factor": None,
                "rule_nominal_concentration_risk_flag": None,
                "rule_volatile_uncertainty_penalty": None,
            },
            "Volatility loss rule is a TODO placeholder until Henry constant and vapor pressure fields are standardized.",
        )


class BioavailabilityRule:
    name = "bioavailability"
    required_inputs = ["logKOC", "OC_or_f_OC"]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        return _stub_output(
            self.name,
            {
                "rule_koc_binding_strength": None,
                "rule_estimated_free_fraction": None,
                "rule_soil_sediment_bioavailability_penalty": None,
            },
            "Soil/sediment organic-carbon bioavailability rule is a TODO placeholder; no free concentration is fabricated.",
        )
