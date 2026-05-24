import math

import pandas as pd

from qsar_dl.rules import RuleOutput, compute_rule_layer, get_rule_registry
from qsar_dl.rules.duration import DurationRule
from qsar_dl.rules.ionization import IonizationRule
from qsar_dl.rules.molecular_weight import MolecularWeightRule
from qsar_dl.rules.route_access import RouteAccessRule
from qsar_dl.rules.solubility import SolubilityRule


def test_rule_output_contract_and_registry():
    output = RuleOutput(features={}, corrections={}, flags={}, explanation="ok")
    assert output.explanation == "ok"
    assert [rule.name for rule in get_rule_registry()]


def test_solubility_normal_missing_and_disabled():
    rule = SolubilityRule()
    normal = rule.compute(
        {"y_pred": 6.0, "MW": 100.0, "water_solubility_mg_l": 0.05},
        {"enabled": True, "no_effects_ratio": 10},
    )
    assert normal.flags["rule_solubility_applicable"] is True
    assert math.isclose(normal.features["rule_solubility_ratio"], 2.0)
    assert normal.features["rule_near_saturation_flag"] == 1
    assert normal.explanation

    missing = rule.compute({"y_pred": 6.0, "MW": 100.0}, {"enabled": True})
    assert missing.flags["rule_solubility_applicable"] is False
    assert "water_solubility_mg_l" in missing.flags["rule_solubility_missing_inputs"]
    assert missing.explanation

    disabled = rule.compute({"y_pred": 6.0, "MW": 100.0, "water_solubility_mg_l": 1.0}, {"enabled": False})
    assert disabled.flags["rule_solubility_applicable"] is False
    assert disabled.flags["rule_solubility_disabled"] is True
    assert disabled.explanation


def test_duration_normal_missing_and_not_applicable():
    rule = DurationRule()
    config = {"enabled": True, "gamma_grid": [0.25], "standard_hours": {"fish_lc50": 96}}
    normal = rule.compute({"duration_h": 24, "endpoint": "LC50", "eco_group": "fish", "target_ptox": 5}, config)
    assert normal.flags["rule_duration_applicable"] is True
    assert math.isclose(normal.features["rule_duration_ratio"], 0.25)
    assert normal.features["rule_short_duration_flag"] == 1
    assert normal.corrections["rule_duration_ptox_adjustment_candidate"] > 0
    assert normal.explanation

    missing = rule.compute({"endpoint": "LC50", "eco_group": "fish"}, config)
    assert missing.flags["rule_duration_applicable"] is False
    assert "duration_h" in missing.flags["rule_duration_missing_inputs"]
    assert missing.explanation

    not_applicable = rule.compute({"duration_h": 48, "endpoint": "LOEC", "eco_group": "fish"}, config)
    assert not_applicable.flags["rule_duration_applicable"] is False
    assert not_applicable.flags["rule_duration_missing_inputs"] == ""
    assert not_applicable.explanation


def test_ionization_normal_missing_and_not_applicable():
    rule = IonizationRule()
    normal = rule.compute({"pH": 8.0, "pKa_acid": 5.0, "logD": 1.2}, {"enabled": True})
    assert normal.flags["rule_ionization_applicable"] is True
    assert normal.features["rule_neutral_fraction"] < 0.01
    assert normal.flags["rule_ionization_flag"] is True
    assert normal.flags["rule_logkow_replaced_by_logd_flag"] is True
    assert normal.explanation

    missing = rule.compute({"pH": 8.0}, {"enabled": True})
    assert missing.flags["rule_ionization_applicable"] is False
    assert "pKa_acid_or_pKa_base" in missing.flags["rule_ionization_missing_inputs"]
    assert missing.explanation

    not_applicable = rule.compute({"ionization_type": "neutral"}, {"enabled": True})
    assert not_applicable.flags["rule_ionization_applicable"] is False
    assert not_applicable.features["rule_neutral_fraction"] == 1.0
    assert not_applicable.explanation


def test_molecular_weight_normal_missing_and_disabled():
    rule = MolecularWeightRule()
    normal = rule.compute({"MW": 800}, {"enabled": True})
    assert normal.flags["rule_molecular_weight_applicable"] is True
    assert math.isclose(normal.features["rule_mw_passive_penalty"], 0.5)
    assert math.isclose(normal.features["rule_passive_uptake_factor"], 0.5)
    assert normal.flags["rule_large_molecule_flag"] is True
    assert normal.explanation

    missing = rule.compute({}, {"enabled": True})
    assert missing.flags["rule_molecular_weight_applicable"] is False
    assert "MW" in missing.flags["rule_molecular_weight_missing_inputs"]
    assert missing.explanation

    disabled = rule.compute({"MW": 800}, {"enabled": False})
    assert disabled.flags["rule_molecular_weight_applicable"] is False
    assert disabled.flags["rule_molecular_weight_disabled"] is True
    assert disabled.explanation


def test_route_access_is_disabled_without_species_route_features():
    rule = RouteAccessRule()
    output = rule.compute({"logKow": 3.0}, {"enabled": True})
    assert output.features["rule_route_access"] is None
    assert output.flags["rule_route_access_applicable"] is False
    assert output.flags["rule_route_access_disabled"] is True
    assert "species_route_features" in output.flags["rule_route_access_missing_inputs"]
    assert output.explanation


def test_compute_rule_layer_returns_feature_table_and_report():
    df = pd.DataFrame(
        [
            {
                "y_pred": 6.0,
                "MW": 100.0,
                "water_solubility_mg_l": 0.05,
                "duration_h": 24,
                "endpoint": "LC50",
                "eco_group": "fish",
                "pH": 8,
                "pKa_acid": 5,
                "logKow": 3,
            },
            {
                "y_pred": 6.0,
                "MW": None,
                "water_solubility_mg_l": 0.05,
                "duration_h": 48,
                "endpoint": "LOEC",
                "eco_group": "fish",
            },
        ]
    )
    table, report = compute_rule_layer(
        df,
        {
            "rules": {
                "solubility": {"enabled": True},
                "duration": {"enabled": True},
                "ionization": {"enabled": True},
                "molecular_weight": {"enabled": True},
            }
        },
    )
    assert len(table) == 2
    assert "rule_solubility_ratio" in table.columns
    assert "rule_duration_ptox_adjustment_candidate" in table.columns
    assert "rule_route_access_disabled" in table.columns
    assert report["n_rows"] == 2
    assert report["rules"]["route_access"]["disabled_count"] == 2
