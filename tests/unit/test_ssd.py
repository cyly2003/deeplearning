from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from qsar_dl.ssd import SSDInsufficientSpeciesError, bootstrap_hc, fit_ssd, plot_ssd, select_sensitive_species


def test_select_sensitive_species_keeps_lowest_positive_value_per_species() -> None:
    data = pd.DataFrame(
        {
            "species": ["Fish A", "Fish A", "Daphnia", "Alga", "Alga", "Bad"],
            "lc50_mg_l": [4.0, 1.5, 0.8, 3.0, 2.0, -1.0],
        }
    )

    selected, report = select_sensitive_species(
        data,
        species_col="species",
        value_col="lc50_mg_l",
        min_species=3,
        return_report=True,
    )

    assert selected["species"].tolist() == ["Daphnia", "Fish A", "Alga"]
    assert selected["lc50_mg_l"].tolist() == [0.8, 1.5, 2.0]
    assert report["n_removed_nonpositive_value"] == 1
    assert report["rejected"] is False


def test_fit_lognormal_hc5_matches_closed_form_estimate() -> None:
    values = np.array([0.8, 1.2, 1.5, 2.0, 2.5, 3.2, 4.0], dtype=float)
    result = fit_ssd(values, distribution="lognormal", min_species=5)

    log_values = np.log(values)
    expected = math.exp(float(np.mean(log_values)) + float(np.std(log_values, ddof=0)) * NormalDist().inv_cdf(0.05))

    assert result.distribution == "lognormal"
    assert result.species_count == 7
    assert math.isclose(result.hc5, expected, rel_tol=1e-12)
    assert math.isclose(result["hc5"], result.hc5)


def test_fit_loglogistic_returns_positive_hc5() -> None:
    values = [0.8, 1.2, 1.5, 2.0, 2.5, 3.2, 4.0]

    result = fit_ssd(values, distribution="loglogistic", min_species=5)

    assert result.distribution == "loglogistic"
    assert result.method in {"scipy_logistic_mle", "numpy_log_moments", "degenerate_log_moments"}
    assert result.hc5 > 0


def test_bootstrap_hc_seed_is_reproducible() -> None:
    values = [0.8, 1.2, 1.5, 2.0, 2.5, 3.2, 4.0]

    first = bootstrap_hc(values, distribution="lognormal", min_species=5, n_bootstrap=100, seed=20260524)
    second = bootstrap_hc(values, distribution="lognormal", min_species=5, n_bootstrap=100, seed=20260524)

    assert first.hc_values == second.hc_values
    assert first.hc5_ci == second.hc5_ci
    assert first.point_estimate == second.point_estimate


def test_fit_ssd_rejects_insufficient_species_with_reason() -> None:
    with pytest.raises(SSDInsufficientSpeciesError) as exc_info:
        fit_ssd([1.0, 2.0, 3.0], min_species=5)

    assert exc_info.value.species_count == 3
    assert exc_info.value.min_species == 5
    assert "at least 5 species" in exc_info.value.reason


def test_plot_ssd_uses_matplotlib_when_available() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    result = fit_ssd([0.8, 1.2, 1.5, 2.0, 2.5, 3.2, 4.0], min_species=5)
    fig, ax = plot_ssd(result)

    assert ax.get_xscale() == "log"
    assert len(ax.lines) >= 2
    assert len(ax.collections) == 1

    matplotlib.pyplot.close(fig)
