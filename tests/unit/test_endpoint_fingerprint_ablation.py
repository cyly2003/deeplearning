import math

import pandas as pd

from run_endpoint_fingerprint_ablation import (
    infer_response_domain,
    parse_endpoint_semantics,
)


def test_lc_endpoint_is_mapped_to_ecx_mortality() -> None:
    row = pd.Series({"endpoint_raw": "LC50", "effect": "MOR", "measurement": "MORT"})

    parsed = parse_endpoint_semantics(row)

    assert parsed["endpoint_task"] == "ecx"
    assert parsed["endpoint_semantic_family"] == "ECx"
    assert parsed["endpoint_source_family"] == "LC"
    assert parsed["response_domain"] == "mortality"
    assert parsed["is_lethal_response"] is True
    assert parsed["effect_percent"] == 50.0
    assert math.isclose(parsed["effect_fraction"], 0.5)


def test_ec_endpoint_keeps_effect_percent_as_continuous_feature() -> None:
    row = pd.Series({"endpoint_raw": "EC10", "effect": "GRO", "measurement": "PGRT"})

    parsed = parse_endpoint_semantics(row)

    assert parsed["endpoint_task"] == "ecx"
    assert parsed["endpoint_source_family"] == "EC"
    assert parsed["response_domain"] == "growth"
    assert parsed["effect_percent"] == 10.0
    assert math.isclose(parsed["effect_fraction"], 0.1)


def test_loec_and_noec_are_threshold_tasks() -> None:
    loec = parse_endpoint_semantics(
        pd.Series({"endpoint_raw": "LOEC", "effect": "REP", "measurement": "PROG"})
    )
    noec = parse_endpoint_semantics(
        pd.Series({"endpoint_raw": "NOEC", "effect": "MOR", "measurement": "SURV"})
    )

    assert loec["endpoint_task"] == "loec"
    assert loec["endpoint_stat_type"] == "threshold_observed_effect"
    assert loec["response_domain"] == "reproduction"
    assert loec["is_chronic_threshold"] is True
    assert noec["endpoint_task"] == "noec"
    assert noec["endpoint_stat_type"] == "threshold_no_observed_effect"
    assert noec["response_domain"] == "mortality"
    assert noec["is_chronic_threshold"] is True


def test_response_domain_falls_back_to_measurement_code() -> None:
    row = pd.Series({"effect": None, "measurement": "IMBL/"})

    assert infer_response_domain(row) == "immobilization"
