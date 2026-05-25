"""Traditional machine-learning baselines for QSAR regression."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from qsar_dl.features.descriptor_groups import (
    build_fixed_group_features,
    load_descriptor_group_dictionary,
)


METRIC_KEYS = ("R2", "RMSE", "MAE", "MAPE")
DEFAULT_MODELS = (
    "pls",
    "elasticnet",
    "svr",
    "random_forest",
    "xgboost",
    "lightgbm",
)
DEFAULT_CONTEXT_COLUMNS = (
    "primary_medium",
    "organism_lifestage",
    "species_ecotox_group",
    "kingdom",
    "phylum_division",
    "class",
    "tax_order",
    "family",
    "genus",
    "endpoint_family",
    "effect_level",
    "duration_h",
)
DEFAULT_EXCLUDED_COLUMNS = {
    "chemical_id",
    "cas_number",
    "dtxsid",
    "smiles",
    "chemical_name",
    "latin_name",
    "split",
    "modeling_split_group",
    "not_modelable_reasons",
    "target_mg_l",
    "target_mol_l",
    "target_unit_family",
}
_SKLEARN_CACHE: dict[str, Any] | None = None
_SKLEARN_IMPORT_ERROR: BaseException | None = None


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature matrix and trace metadata for a baseline feature set."""

    X: pd.DataFrame
    y: pd.Series
    feature_set: str
    feature_columns: list[str]
    source_columns: list[str]
    skipped_columns: list[str]
    skipped_context_columns: list[str]
    missing_descriptor_columns: list[str]
    dropped_rows: int


@dataclass(frozen=True)
class TrainingResult:
    """Estimator training result, including optional-dependency skips."""

    model_name: str
    estimator: Any | None
    status: str
    skip_reason: str | None
    params: dict[str, Any]


def build_feature_matrix(
    data: pd.DataFrame,
    *,
    target_column: str = "target_ptox",
    feature_set: str = "standard",
    feature_columns: Sequence[str] | None = None,
    context_columns: Sequence[str] | None = None,
    species_context_columns: Sequence[str] | None = None,
    descriptor_group_dict: Mapping[str, Any] | None = None,
    descriptor_group_path: str | Path | None = None,
    chemical_id_column: str = "chemical_id",
) -> FeatureMatrix:
    """Build a numeric baseline matrix from standard or fixed group features.

    Missing requested context columns are reported in metadata and otherwise
    ignored so the same config can run before species/context modules are ready.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if target_column not in data.columns:
        raise ValueError(f"target column not found: {target_column}")

    target = pd.to_numeric(data[target_column], errors="coerce")
    valid_target = target.notna()
    dropped_rows = int((~valid_target).sum())
    working = data.loc[valid_target].copy()
    y = target.loc[valid_target].astype("float64")
    if working.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    normalized_feature_set = _normalize_feature_set(feature_set)
    requested_context = _requested_context_columns(context_columns, species_context_columns)

    if normalized_feature_set == "standard":
        base_frame, source_columns, skipped_columns = _build_standard_frame(
            working,
            target_column=target_column,
            feature_columns=feature_columns,
            context_columns=requested_context,
        )
        missing_descriptor_columns: list[str] = []
    elif normalized_feature_set == "fixed_descriptor_groups":
        base_frame, source_columns, skipped_columns, missing_descriptor_columns = (
            _build_fixed_descriptor_group_frame(
                working,
                descriptor_group_dict=descriptor_group_dict,
                descriptor_group_path=descriptor_group_path,
                chemical_id_column=chemical_id_column,
            )
        )
    else:  # pragma: no cover - guarded by _normalize_feature_set.
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    context_frame, included_context, skipped_context = _build_context_frame(
        working, requested_context, excluded_columns=set(source_columns)
    )
    frames = [frame for frame in (base_frame, context_frame) if not frame.empty]
    if not frames:
        raise ValueError(f"No usable features were found for feature_set={feature_set!r}.")

    X = pd.concat(frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated()].astype("float64")
    return FeatureMatrix(
        X=X,
        y=y.loc[X.index],
        feature_set=normalized_feature_set,
        feature_columns=list(X.columns),
        source_columns=source_columns + included_context,
        skipped_columns=sorted(set(skipped_columns + skipped_context)),
        skipped_context_columns=skipped_context,
        missing_descriptor_columns=missing_descriptor_columns,
        dropped_rows=dropped_rows,
    )


def train_regressor(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    random_state: int = 20260524,
    model_params: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train one supported regressor or return a skip for missing optional deps."""

    if X_train.shape[1] == 0:
        raise ValueError("X_train must contain at least one feature.")
    if len(X_train) < 2:
        raise ValueError("At least two training rows are required.")

    params = dict(model_params or {})
    normalized_name = _normalize_model_name(model_name)
    estimator: Any
    sklearn = _load_sklearn()
    if sklearn is None:
        return TrainingResult(
            model_name=normalized_name,
            estimator=None,
            status="skipped",
            skip_reason=_optional_dependency_reason("scikit-learn", _SKLEARN_IMPORT_ERROR),
            params=params,
        )

    if normalized_name == "pls":
        requested_components = int(params.pop("n_components", 2))
        n_components = _safe_pls_components(
            requested_components=requested_components,
            n_samples=len(X_train),
            n_features=X_train.shape[1],
        )
        estimator = sklearn["Pipeline"](
            [
                ("imputer", sklearn["SimpleImputer"](strategy=params.pop("imputer_strategy", "median"))),
                ("scaler", sklearn["StandardScaler"]()),
                ("model", sklearn["PLSRegression"](n_components=n_components, **params)),
            ]
        )
    elif normalized_name == "elasticnet":
        params.setdefault("alpha", 0.1)
        params.setdefault("l1_ratio", 0.5)
        params.setdefault("max_iter", 10000)
        params.setdefault("random_state", random_state)
        estimator = sklearn["Pipeline"](
            [
                ("imputer", sklearn["SimpleImputer"](strategy=params.pop("imputer_strategy", "median"))),
                ("scaler", sklearn["StandardScaler"]()),
                ("model", sklearn["ElasticNet"](**params)),
            ]
        )
    elif normalized_name == "svr":
        params.setdefault("kernel", "rbf")
        params.setdefault("C", 1.0)
        params.setdefault("epsilon", 0.1)
        estimator = sklearn["Pipeline"](
            [
                ("imputer", sklearn["SimpleImputer"](strategy=params.pop("imputer_strategy", "median"))),
                ("scaler", sklearn["StandardScaler"]()),
                ("model", sklearn["SVR"](**params)),
            ]
        )
    elif normalized_name == "random_forest":
        params.setdefault("n_estimators", 200)
        params.setdefault("min_samples_leaf", 1)
        params.setdefault("random_state", random_state)
        estimator = sklearn["Pipeline"](
            [
                ("imputer", sklearn["SimpleImputer"](strategy=params.pop("imputer_strategy", "median"))),
                ("model", sklearn["RandomForestRegressor"](**params)),
            ]
        )
    elif normalized_name == "xgboost":
        optional = _make_xgboost_regressor(params, random_state)
        if optional.status == "skipped":
            return optional
        estimator = optional.estimator
    elif normalized_name == "lightgbm":
        optional = _make_lightgbm_regressor(params, random_state)
        if optional.status == "skipped":
            return optional
        estimator = optional.estimator
    else:  # pragma: no cover - guarded by _normalize_model_name.
        raise ValueError(f"Unsupported model_name: {model_name}")

    estimator.fit(X_train, np.asarray(y_train, dtype="float64"))
    return TrainingResult(
        model_name=normalized_name,
        estimator=estimator,
        status="trained",
        skip_reason=None,
        params=params,
    )


def evaluate_regression(
    y_true: pd.Series | np.ndarray | Sequence[float],
    y_pred: pd.Series | np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Return standard regression metrics for pTox baselines."""

    true = np.asarray(y_true, dtype="float64").reshape(-1)
    pred = np.asarray(y_pred, dtype="float64").reshape(-1)
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    finite = np.isfinite(true) & np.isfinite(pred)
    if not finite.any():
        return _empty_metrics()

    true = true[finite]
    pred = pred[finite]
    mape_mask = np.abs(true) > 1.0e-12
    return {
        "R2": _r2_score(true, pred),
        "RMSE": float(math.sqrt(np.mean((true - pred) ** 2))),
        "MAE": float(np.mean(np.abs(true - pred))),
        "MAPE": float(np.mean(np.abs((true[mape_mask] - pred[mape_mask]) / true[mape_mask])) * 100.0)
        if mape_mask.any()
        else math.nan,
    }


def run_baseline_experiment(
    data: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run configured baseline feature sets and regressors on a DataFrame."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    config = dict(config or {})
    baseline_config = dict(config.get("baseline_ml", config.get("baseline", {})))
    target_column = _target_column(config)
    random_state = _random_state(config, baseline_config)
    models = list(baseline_config.get("models", DEFAULT_MODELS))
    test_size = float(baseline_config.get("test_size", 0.25))
    model_params = dict(baseline_config.get("model_params", {}))

    output: dict[str, Any] = {
        "target_column": target_column,
        "random_state": random_state,
        "feature_sets": {},
    }

    for feature_config in _feature_set_configs(baseline_config):
        feature_name = str(feature_config.get("name", feature_config.get("type", "standard")))
        matrix = build_feature_matrix(
            data,
            target_column=target_column,
            feature_set=str(feature_config.get("type", feature_name)),
            feature_columns=feature_config.get("feature_columns"),
            context_columns=feature_config.get("context_columns", baseline_config.get("context_columns")),
            species_context_columns=feature_config.get(
                "species_context_columns",
                baseline_config.get("species_context_columns"),
            ),
            descriptor_group_dict=_descriptor_group_dict(feature_config, baseline_config, config),
            descriptor_group_path=feature_config.get(
                "descriptor_group_path",
                baseline_config.get("descriptor_group_path"),
            ),
            chemical_id_column=str(
                feature_config.get(
                    "chemical_id_column",
                    baseline_config.get("chemical_id_column", "chemical_id"),
                )
            ),
        )
        X_train, X_test, y_train, y_test = _split_xy(
            data.loc[matrix.X.index],
            matrix.X,
            matrix.y,
            split_column=str(baseline_config.get("split_column", "split")),
            train_values=baseline_config.get("train_values", ("train",)),
            test_values=baseline_config.get("test_values", ("test", "validation", "val")),
            test_size=test_size,
            random_state=random_state,
        )

        model_results: dict[str, dict[str, Any]] = {}
        for model_name in models:
            normalized_name = _normalize_model_name(str(model_name))
            trained = train_regressor(
                normalized_name,
                X_train,
                y_train,
                random_state=random_state,
                model_params=model_params.get(normalized_name, {}),
            )
            if trained.status == "skipped":
                model_results[normalized_name] = {
                    "status": trained.status,
                    "metrics": _empty_metrics(),
                    "skip_reason": trained.skip_reason,
                }
                continue

            predictions = np.asarray(trained.estimator.predict(X_test), dtype="float64").reshape(-1)
            model_results[normalized_name] = {
                "status": "trained",
                "metrics": evaluate_regression(y_test, predictions),
                "skip_reason": None,
            }

        output["feature_sets"][feature_name] = {
            "feature_set": matrix.feature_set,
            "n_features": int(matrix.X.shape[1]),
            "feature_columns": matrix.feature_columns,
            "source_columns": matrix.source_columns,
            "skipped_columns": matrix.skipped_columns,
            "skipped_context_columns": matrix.skipped_context_columns,
            "missing_descriptor_columns": matrix.missing_descriptor_columns,
            "dropped_rows": matrix.dropped_rows,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "models": model_results,
        }

    return output


def _build_standard_frame(
    data: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str] | None,
    context_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if feature_columns is None:
        excluded = set(DEFAULT_EXCLUDED_COLUMNS)
        excluded.add(target_column)
        excluded.update(context_columns)
        source_columns = [
            column
            for column in data.columns
            if column not in excluded
            and not (column.startswith("target_") and column != target_column)
            and (is_numeric_dtype(data[column]) or is_bool_dtype(data[column]))
        ]
        skipped_columns: list[str] = []
    else:
        requested = _dedupe_preserve_order([str(column) for column in feature_columns])
        source_columns = [column for column in requested if column in data.columns]
        skipped_columns = [column for column in requested if column not in data.columns]

    return _encode_mixed_frame(data[source_columns]), source_columns, skipped_columns


def _build_fixed_descriptor_group_frame(
    data: pd.DataFrame,
    *,
    descriptor_group_dict: Mapping[str, Any] | None,
    descriptor_group_path: str | Path | None,
    chemical_id_column: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if descriptor_group_dict is None and descriptor_group_path is None:
        group_columns = [
            column
            for column in data.columns
            if column.startswith("desc_group_") and is_numeric_dtype(data[column])
        ]
        if not group_columns:
            raise ValueError(
                "fixed_descriptor_groups requires descriptor_group_dict, "
                "descriptor_group_path, or precomputed desc_group_* columns."
            )
        return _encode_mixed_frame(data[group_columns]), group_columns, [], []

    group_dict = (
        dict(descriptor_group_dict)
        if descriptor_group_dict is not None
        else load_descriptor_group_dictionary(Path(descriptor_group_path))
    )
    if chemical_id_column not in data.columns:
        raise ValueError(
            f"fixed_descriptor_groups requires a '{chemical_id_column}' column "
            "for descriptor group traceability."
        )

    descriptor_columns = _descriptor_columns_from_group_dict(group_dict)
    present_descriptors = [column for column in descriptor_columns if column in data.columns]
    missing_descriptors = [column for column in descriptor_columns if column not in data.columns]
    descriptor_df = data[[chemical_id_column] + present_descriptors].rename(
        columns={chemical_id_column: "chemical_id"}
    )
    group_features = build_fixed_group_features(descriptor_df, group_dict)
    group_feature_columns = [
        column for column in group_features.columns if column != "chemical_id"
    ]
    return (
        _encode_mixed_frame(group_features[group_feature_columns]).set_index(data.index),
        present_descriptors,
        [],
        missing_descriptors,
    )


def _build_context_frame(
    data: pd.DataFrame,
    context_columns: Sequence[str],
    *,
    excluded_columns: set[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    requested = _dedupe_preserve_order([str(column) for column in context_columns])
    included = [
        column
        for column in requested
        if column in data.columns and column not in excluded_columns
    ]
    skipped = [column for column in requested if column not in data.columns]
    return _encode_mixed_frame(data[included]), included, skipped


def _encode_mixed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)

    numeric_columns = [
        column
        for column in frame.columns
        if is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column])
    ]
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]

    parts: list[pd.DataFrame] = []
    if numeric_columns:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        parts.append(numeric.astype("float64"))
    if categorical_columns:
        categorical = frame[categorical_columns].astype("string").fillna("__missing__")
        parts.append(pd.get_dummies(categorical, prefix=categorical_columns, dtype="float64"))

    if not parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(parts, axis=1)


def _split_xy(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split_column: str,
    train_values: Sequence[str],
    test_values: Sequence[str],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if split_column in data.columns:
        split_values = data[split_column].astype("string")
        train_mask = split_values.isin([str(value) for value in train_values])
        test_mask = split_values.isin([str(value) for value in test_values])
        if train_mask.any() and test_mask.any():
            return (
                X.loc[train_mask],
                X.loc[test_mask],
                y.loc[train_mask],
                y.loc[test_mask],
            )

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 when no split column is usable.")
    if len(X) < 2:
        raise ValueError("At least two rows are required for a random train/test split.")

    rng = np.random.default_rng(random_state)
    positions = rng.permutation(len(X))
    test_count = int(math.ceil(len(X) * test_size))
    test_count = min(max(test_count, 1), len(X) - 1)
    test_positions = positions[:test_count]
    train_positions = positions[test_count:]
    return (
        X.iloc[train_positions],
        X.iloc[test_positions],
        y.iloc[train_positions],
        y.iloc[test_positions],
    )


def _make_xgboost_regressor(
    params: dict[str, Any], random_state: int
) -> TrainingResult:
    sklearn = _load_sklearn()
    if sklearn is None:
        return TrainingResult(
            model_name="xgboost",
            estimator=None,
            status="skipped",
            skip_reason=_optional_dependency_reason("scikit-learn", _SKLEARN_IMPORT_ERROR),
            params=params,
        )
    if importlib.util.find_spec("xgboost") is None:
        return TrainingResult(
            model_name="xgboost",
            estimator=None,
            status="skipped",
            skip_reason="Optional dependency 'xgboost' is not installed.",
            params=params,
        )
    try:
        from xgboost import XGBRegressor  # type: ignore[import-not-found]
    except Exception as exc:
        return TrainingResult(
            model_name="xgboost",
            estimator=None,
            status="skipped",
            skip_reason=_optional_dependency_reason("xgboost", exc),
            params=params,
        )

    params.setdefault("n_estimators", 200)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("max_depth", 4)
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("random_state", random_state)
    params.setdefault("verbosity", 0)
    estimator = sklearn["Pipeline"](
        [
            ("imputer", sklearn["SimpleImputer"](strategy=params.pop("imputer_strategy", "median"))),
            ("model", XGBRegressor(**params)),
        ]
    )
    return TrainingResult("xgboost", estimator, "trained", None, params)


def _make_lightgbm_regressor(
    params: dict[str, Any], random_state: int
) -> TrainingResult:
    sklearn = _load_sklearn()
    if sklearn is None:
        return TrainingResult(
            model_name="lightgbm",
            estimator=None,
            status="skipped",
            skip_reason=_optional_dependency_reason("scikit-learn", _SKLEARN_IMPORT_ERROR),
            params=params,
        )
    if importlib.util.find_spec("lightgbm") is None:
        return TrainingResult(
            model_name="lightgbm",
            estimator=None,
            status="skipped",
            skip_reason="Optional dependency 'lightgbm' is not installed.",
            params=params,
        )
    try:
        from lightgbm import LGBMRegressor  # type: ignore[import-not-found]
    except Exception as exc:
        return TrainingResult(
            model_name="lightgbm",
            estimator=None,
            status="skipped",
            skip_reason=_optional_dependency_reason("lightgbm", exc),
            params=params,
        )

    params.setdefault("n_estimators", 200)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("random_state", random_state)
    params.setdefault("verbosity", -1)
    estimator = sklearn["Pipeline"](
        [
            ("imputer", sklearn["SimpleImputer"](strategy=params.pop("imputer_strategy", "median"))),
            ("model", LGBMRegressor(**params)),
        ]
    )
    return TrainingResult("lightgbm", estimator, "trained", None, params)


def _feature_set_configs(baseline_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_configs = baseline_config.get(
        "feature_sets",
        (
            {"name": "standard", "type": "standard"},
            {"name": "fixed_descriptor_groups", "type": "fixed_descriptor_groups"},
        ),
    )
    configs: list[dict[str, Any]] = []
    for item in raw_configs:
        if isinstance(item, str):
            configs.append({"name": item, "type": item})
        elif isinstance(item, Mapping):
            configs.append(dict(item))
        else:
            raise TypeError("baseline_ml.feature_sets must contain strings or mappings.")
    return configs


def _descriptor_group_dict(
    feature_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for candidate in (
        feature_config.get("descriptor_group_dict"),
        feature_config.get("descriptor_groups"),
        baseline_config.get("descriptor_group_dict"),
        baseline_config.get("descriptor_groups"),
    ):
        if isinstance(candidate, Mapping):
            return candidate
    if {"descriptor_source", "standardization", "groups"}.issubset(config):
        return {
            "descriptor_source": config["descriptor_source"],
            "standardization": config["standardization"],
            "groups": config["groups"],
        }
    return None


def _target_column(config: Mapping[str, Any]) -> str:
    if "target_column" in config:
        return str(config["target_column"])
    model_config = config.get("model")
    if isinstance(model_config, Mapping) and "target_column" in model_config:
        return str(model_config["target_column"])
    target_config = config.get("target")
    if isinstance(target_config, Mapping) and "column" in target_config:
        return str(target_config["column"])
    return "target_ptox"


def _random_state(config: Mapping[str, Any], baseline_config: Mapping[str, Any]) -> int:
    if "random_state" in baseline_config:
        return int(baseline_config["random_state"])
    experiment_config = config.get("experiment")
    if isinstance(experiment_config, Mapping) and "seed" in experiment_config:
        return int(experiment_config["seed"])
    return 20260524


def _requested_context_columns(
    context_columns: Sequence[str] | None,
    species_context_columns: Sequence[str] | None,
) -> list[str]:
    if context_columns is None and species_context_columns is None:
        return list(DEFAULT_CONTEXT_COLUMNS)
    requested: list[str] = []
    if context_columns is not None:
        requested.extend(str(column) for column in context_columns)
    if species_context_columns is not None:
        requested.extend(str(column) for column in species_context_columns)
    return _dedupe_preserve_order(requested)


def _descriptor_columns_from_group_dict(group_dict: Mapping[str, Any]) -> list[str]:
    groups = group_dict.get("groups", {})
    descriptors: list[str] = []
    if isinstance(groups, Mapping):
        for group_config in groups.values():
            if isinstance(group_config, Mapping) and isinstance(
                group_config.get("descriptors"), Mapping
            ):
                descriptors.extend(str(column) for column in group_config["descriptors"])
    return _dedupe_preserve_order(descriptors)


def _normalize_feature_set(feature_set: str) -> str:
    normalized = str(feature_set).strip().lower().replace("-", "_")
    aliases = {
        "standard_features": "standard",
        "raw_descriptors": "standard",
        "fixed_descriptor_group_features": "fixed_descriptor_groups",
        "fixed_group_features": "fixed_descriptor_groups",
        "descriptor_groups": "fixed_descriptor_groups",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"standard", "fixed_descriptor_groups"}:
        raise ValueError(
            "feature_set must be one of: standard, fixed_descriptor_groups."
        )
    return normalized


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower().replace("-", "_")
    aliases = {
        "elastic_net": "elasticnet",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "plsregression": "pls",
        "pls_regression": "pls",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"pls", "elasticnet", "svr", "random_forest", "xgboost", "lightgbm"}
    if normalized not in supported:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return normalized


def _safe_pls_components(
    *, requested_components: int, n_samples: int, n_features: int
) -> int:
    if requested_components < 1:
        raise ValueError("PLS n_components must be >= 1.")
    return max(1, min(requested_components, n_features, n_samples - 1))


def _empty_metrics() -> dict[str, float]:
    return {key: math.nan for key in METRIC_KEYS}


def _load_sklearn() -> dict[str, Any] | None:
    """Load scikit-learn lazily so a broken optional stack does not break imports."""

    global _SKLEARN_CACHE, _SKLEARN_IMPORT_ERROR
    if _SKLEARN_CACHE is not None:
        return _SKLEARN_CACHE
    if _SKLEARN_IMPORT_ERROR is not None:
        return None

    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except Exception as exc:
        _SKLEARN_IMPORT_ERROR = exc
        return None

    _SKLEARN_CACHE = {
        "PLSRegression": PLSRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "ElasticNet": ElasticNet,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
    }
    return _SKLEARN_CACHE


def _optional_dependency_reason(package: str, exc: BaseException | None) -> str:
    if exc is None:
        return f"Optional dependency '{package}' is not installed or unavailable."
    return f"Optional dependency '{package}' is unavailable: {type(exc).__name__}: {exc}"


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
