"""Species sensitivity distribution fitting and plotting.

The helpers in this module are intentionally small and dependency-light. The
lognormal model uses closed-form log-scale estimates. The loglogistic model
uses scipy's logistic MLE on log concentrations when scipy is available, and a
NumPy moment approximation otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

DistributionName = Literal["lognormal", "loglogistic"]

_NORMAL = NormalDist()


class SSDInsufficientSpeciesError(ValueError):
    """Raised when an SSD fit is rejected because too few species are available."""

    def __init__(self, species_count: int, min_species: int, reason: str) -> None:
        self.species_count = species_count
        self.min_species = min_species
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class SSDFitResult:
    """Fitted SSD model and hazardous concentration estimate."""

    distribution: DistributionName
    method: str
    species_count: int
    params: Mapping[str, float]
    hc5: float
    hc_p: float
    values: tuple[float, ...]
    plotting_positions: tuple[float, ...]

    def quantile(self, probability: float) -> float:
        """Return the fitted concentration quantile for a cumulative probability."""

        _validate_probability(probability, "probability")
        return _quantile(self.distribution, self.params, probability)

    def cdf(self, concentrations: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return fitted cumulative probabilities for concentrations."""

        return _cdf(self.distribution, self.params, np.asarray(concentrations, dtype=float))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation for reports and tests."""

        return {
            "distribution": self.distribution,
            "method": self.method,
            "species_count": self.species_count,
            "params": dict(self.params),
            "hc5": self.hc5,
            "hc_p": self.hc_p,
            "values": list(self.values),
            "plotting_positions": list(self.plotting_positions),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class BootstrapHCResult:
    """Bootstrap hazardous concentration summary."""

    distribution: DistributionName
    method: str
    hc_p: float
    confidence_level: float
    random_seed: int | None
    n_bootstrap: int
    point_estimate: float
    ci_low: float
    ci_high: float
    hc_values: tuple[float, ...]
    fit_failures: int = 0

    @property
    def hc5_ci(self) -> tuple[float, float]:
        """Return the percentile confidence interval."""

        return (self.ci_low, self.ci_high)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation for reports and tests."""

        return {
            "distribution": self.distribution,
            "method": self.method,
            "hc_p": self.hc_p,
            "confidence_level": self.confidence_level,
            "random_seed": self.random_seed,
            "n_bootstrap": self.n_bootstrap,
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "hc5_ci": self.hc5_ci,
            "hc_values": list(self.hc_values),
            "fit_failures": self.fit_failures,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def select_sensitive_species(
    data: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    species_col: str = "species",
    value_col: str = "toxicity",
    keep: Literal["min", "max"] = "min",
    min_species: int | None = None,
    positive_only: bool = True,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Select one toxicity value per species for SSD fitting.

    By default this keeps the minimum positive toxicity concentration per
    species, which corresponds to the most sensitive observation when lower
    concentrations indicate stronger toxicity. A cleaning report can be returned
    so dropped rows and insufficient-species decisions are traceable.
    """

    if keep not in {"min", "max"}:
        raise ValueError("keep must be either 'min' or 'max'.")

    df = pd.DataFrame(data).copy()
    missing_columns = [col for col in (species_col, value_col) if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required SSD column(s): {', '.join(missing_columns)}.")

    report: dict[str, Any] = {
        "species_col": species_col,
        "value_col": value_col,
        "keep": keep,
        "positive_only": positive_only,
        "n_input_rows": int(len(df)),
    }

    species_mask = df[species_col].notna() & df[species_col].astype(str).str.strip().ne("")
    report["n_removed_missing_species"] = int((~species_mask).sum())
    clean = df.loc[species_mask].copy()

    numeric_values = pd.to_numeric(clean[value_col], errors="coerce")
    value_mask = numeric_values.notna()
    report["n_removed_missing_or_non_numeric_value"] = int((~value_mask).sum())
    clean = clean.loc[value_mask].copy()
    numeric_values = numeric_values.loc[value_mask].astype(float)

    finite_mask = pd.Series(np.isfinite(numeric_values.to_numpy()), index=clean.index)
    report["n_removed_nonfinite_value"] = int((~finite_mask).sum())
    clean = clean.loc[finite_mask].copy()
    numeric_values = numeric_values.loc[finite_mask].astype(float)

    if positive_only:
        positive_mask = numeric_values > 0
        report["n_removed_nonpositive_value"] = int((~positive_mask).sum())
        clean = clean.loc[positive_mask].copy()
        numeric_values = numeric_values.loc[positive_mask].astype(float)
    else:
        report["n_removed_nonpositive_value"] = 0

    clean[value_col] = numeric_values.to_numpy(dtype=float)
    report["n_valid_rows"] = int(len(clean))

    if clean.empty:
        selected = clean.reset_index(drop=True)
    else:
        grouped = clean.groupby(species_col, sort=True, dropna=False)[value_col]
        selected_index = grouped.idxmin() if keep == "min" else grouped.idxmax()
        selected = clean.loc[selected_index].sort_values(value_col, ascending=True).reset_index(drop=True)

    report["n_selected_species"] = int(selected[species_col].nunique(dropna=True))
    report["min_species"] = min_species
    if min_species is not None and report["n_selected_species"] < min_species:
        report["rejected"] = True
        report["reason"] = (
            f"SSD fitting requires at least {min_species} species, "
            f"but only {report['n_selected_species']} valid species were selected."
        )
    else:
        report["rejected"] = False
        report["reason"] = ""

    if return_report:
        return selected, report
    return selected


def fit_ssd(
    data: pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    distribution: str = "lognormal",
    species_col: str | None = None,
    value_col: str | None = None,
    min_species: int = 5,
    hc_p: float = 0.05,
) -> SSDFitResult:
    """Fit a lognormal or loglogistic SSD and return an HC estimate.

    Parameters
    ----------
    data:
        Either a numeric sequence of one selected toxicity value per species, or
        a DataFrame containing species and toxicity columns.
    distribution:
        ``"lognormal"`` or ``"loglogistic"``.
    species_col, value_col:
        DataFrame columns. When both are provided, the most sensitive positive
        value per species is selected before fitting.
    min_species:
        Minimum number of valid species required for fitting.
    hc_p:
        Hazardous concentration cumulative probability. ``0.05`` gives HC5.
    """

    model = _normalize_distribution(distribution)
    _validate_probability(hc_p, "hc_p")
    if min_species < 1:
        raise ValueError("min_species must be at least 1.")

    values = _extract_values(data, species_col=species_col, value_col=value_col, min_species=min_species)
    species_count = int(values.size)
    if species_count < min_species:
        reason = f"SSD fitting rejected: at least {min_species} species are required, but {species_count} were available."
        raise SSDInsufficientSpeciesError(species_count, min_species, reason)

    log_values = np.log(values)
    if model == "lognormal":
        params = _fit_lognormal(log_values)
        method = "closed_form_log_mle"
    else:
        params, method = _fit_loglogistic(log_values)

    hc = _quantile(model, params, hc_p)
    return SSDFitResult(
        distribution=model,
        method=method,
        species_count=species_count,
        params=params,
        hc5=hc,
        hc_p=hc_p,
        values=tuple(float(value) for value in np.sort(values)),
        plotting_positions=tuple(float(value) for value in _plotting_positions(species_count)),
    )


def bootstrap_hc(
    data: SSDFitResult | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    distribution: str = "lognormal",
    species_col: str | None = None,
    value_col: str | None = None,
    min_species: int = 5,
    hc_p: float = 0.05,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapHCResult:
    """Bootstrap HC estimates by resampling selected species values."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    _validate_probability(confidence_level, "confidence_level")

    if isinstance(data, SSDFitResult):
        base_fit = data
        values = np.asarray(data.values, dtype=float)
        model = data.distribution
    else:
        model = _normalize_distribution(distribution)
        values = _extract_values(data, species_col=species_col, value_col=value_col, min_species=min_species)
        base_fit = fit_ssd(values, distribution=model, min_species=min_species, hc_p=hc_p)

    if values.size < min_species:
        reason = f"SSD bootstrap rejected: at least {min_species} species are required, but {values.size} were available."
        raise SSDInsufficientSpeciesError(int(values.size), min_species, reason)

    rng = np.random.default_rng(seed)
    hc_values: list[float] = []
    fit_failures = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        try:
            hc_values.append(fit_ssd(sample, distribution=model, min_species=min_species, hc_p=hc_p).hc5)
        except ValueError:
            fit_failures += 1

    if not hc_values:
        raise ValueError("All SSD bootstrap fits failed; inspect input values and distribution choice.")

    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(np.asarray(hc_values, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapHCResult(
        distribution=model,
        method=base_fit.method,
        hc_p=hc_p,
        confidence_level=confidence_level,
        random_seed=seed,
        n_bootstrap=n_bootstrap,
        point_estimate=base_fit.hc5,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        hc_values=tuple(float(value) for value in hc_values),
        fit_failures=fit_failures,
    )


def plot_ssd(
    fit: SSDFitResult,
    *,
    ax: Any | None = None,
    title: str | None = None,
    xlabel: str = "Toxicity concentration",
    ylabel: str = "Cumulative probability",
    show_hc: bool = True,
    n_grid: int = 200,
    apply_publication_style: bool = True,
) -> tuple[Any, Any]:
    """Plot empirical species sensitivities and the fitted SSD curve."""

    if n_grid < 20:
        raise ValueError("n_grid must be at least 20.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        message = (
            "matplotlib is required for plot_ssd. Install the visualization extra, "
            "for example: pip install -e .[viz]"
        )
        raise ImportError(message) from exc

    if apply_publication_style:
        try:
            from qsar_dl.visualization import get_palette, set_publication_style

            set_publication_style()
            colors = get_palette("journal", n_colors=3)
        except ImportError:
            colors = ["#0072B2", "#D55E00", "#009E73"]
    else:
        colors = ["#0072B2", "#D55E00", "#009E73"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
    else:
        fig = ax.figure

    values = np.asarray(fit.values, dtype=float)
    lower = min(values.min(), fit.hc5) * 0.8
    upper = max(values.max(), fit.hc5) * 1.2
    if lower <= 0:
        lower = values.min() * 0.5
    grid = np.exp(np.linspace(math.log(lower), math.log(upper), n_grid))

    ax.scatter(values, fit.plotting_positions, color=colors[0], edgecolor="black", linewidth=0.6, label="Selected species")
    ax.plot(grid, fit.cdf(grid), color=colors[1], linewidth=2.0, label=f"{fit.distribution} fit")
    if show_hc:
        ax.axvline(fit.hc5, color=colors[2], linestyle="--", linewidth=1.5, label=f"HC{int(round(fit.hc_p * 100))}")
        ax.axhline(fit.hc_p, color=colors[2], linestyle=":", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"SSD {fit.distribution}")
    ax.legend()
    return fig, ax


def _normalize_distribution(distribution: str) -> DistributionName:
    key = distribution.strip().lower().replace("-", "").replace("_", "")
    if key == "lognormal":
        return "lognormal"
    if key == "loglogistic":
        return "loglogistic"
    raise ValueError("distribution must be one of: lognormal, loglogistic.")


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _extract_values(
    data: pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    species_col: str | None,
    value_col: str | None,
    min_species: int,
) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        if value_col is None:
            raise ValueError("value_col is required when fitting SSD from a DataFrame.")
        if species_col is not None:
            selected, report = select_sensitive_species(
                data,
                species_col=species_col,
                value_col=value_col,
                min_species=min_species,
                return_report=True,
            )
            if report["rejected"]:
                raise SSDInsufficientSpeciesError(
                    int(report["n_selected_species"]),
                    min_species,
                    str(report["reason"]),
                )
            raw_values = selected[value_col].to_numpy(dtype=float)
        else:
            raw_values = pd.to_numeric(data[value_col], errors="coerce").to_numpy(dtype=float)
    else:
        raw_values = np.asarray(data, dtype=float)

    values = np.ravel(raw_values).astype(float)
    invalid_mask = ~np.isfinite(values) | (values <= 0)
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        raise ValueError(f"SSD values must be positive and finite; found {n_invalid} invalid value(s).")
    return values


def _fit_lognormal(log_values: np.ndarray) -> dict[str, float]:
    mu = float(np.mean(log_values))
    sigma = float(np.std(log_values, ddof=0))
    return {"mu": mu, "sigma": sigma}


def _fit_loglogistic(log_values: np.ndarray) -> tuple[dict[str, float], str]:
    if float(np.std(log_values, ddof=0)) == 0.0:
        return {"loc": float(np.mean(log_values)), "scale": 0.0}, "degenerate_log_moments"

    try:
        from scipy import stats
    except ImportError:
        loc = float(np.mean(log_values))
        scale = float(np.std(log_values, ddof=0) * math.sqrt(3.0) / math.pi)
        return {"loc": loc, "scale": scale}, "numpy_log_moments"

    loc, scale = stats.logistic.fit(log_values)
    return {"loc": float(loc), "scale": float(scale)}, "scipy_logistic_mle"


def _plotting_positions(n_values: int) -> np.ndarray:
    ranks = np.arange(1, n_values + 1, dtype=float)
    return (ranks - 0.5) / n_values


def _quantile(distribution: DistributionName, params: Mapping[str, float], probability: float) -> float:
    _validate_probability(probability, "probability")
    if distribution == "lognormal":
        mu = float(params["mu"])
        sigma = float(params["sigma"])
        return float(math.exp(mu + sigma * _NORMAL.inv_cdf(probability)))

    loc = float(params["loc"])
    scale = float(params["scale"])
    if scale == 0.0:
        return float(math.exp(loc))
    return float(math.exp(loc + scale * _logit(probability)))


def _cdf(distribution: DistributionName, params: Mapping[str, float], concentrations: np.ndarray) -> np.ndarray:
    values = np.asarray(concentrations, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    positive_mask = values > 0
    if not positive_mask.any():
        return output

    log_values = np.log(values[positive_mask])
    if distribution == "lognormal":
        mu = float(params["mu"])
        sigma = float(params["sigma"])
        if sigma == 0.0:
            output[positive_mask] = (log_values >= mu).astype(float)
        else:
            output[positive_mask] = [_NORMAL.cdf(float((value - mu) / sigma)) for value in log_values]
    else:
        loc = float(params["loc"])
        scale = float(params["scale"])
        if scale == 0.0:
            output[positive_mask] = (log_values >= loc).astype(float)
        else:
            output[positive_mask] = 1.0 / (1.0 + np.exp(-((log_values - loc) / scale)))
    return output


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))
