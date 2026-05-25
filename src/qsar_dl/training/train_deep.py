"""Deep-training helpers for residual QSAR smoke tests and real baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from qsar_dl.training.baseline_ml import evaluate_regression
from qsar_dl.models.residual_qsar import ResidualQSARModel

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - depends on optional install.
    torch = None
    DataLoader = None
    Dataset = object
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


DEFAULT_DESCRIPTOR_COLUMNS = (
    "rdkit_descriptor_mol_wt",
    "rdkit_descriptor_exact_mol_wt",
    "rdkit_descriptor_mol_logp",
    "rdkit_descriptor_tpsa",
    "rdkit_descriptor_h_bond_donors",
    "rdkit_descriptor_h_bond_acceptors",
    "rdkit_descriptor_rotatable_bonds",
    "rdkit_descriptor_ring_count",
    "rdkit_descriptor_heavy_atom_count",
    "rdkit_descriptor_fraction_csp3",
    "rdkit_descriptor_formal_charge",
    "molecular_weight_g_mol",
    "logkow",
    "molecular_weight_g_mol_missing_flag",
    "logkow_missing_flag",
)
DEFAULT_FINGERPRINT_PREFIX = "morgan_fp_"


@dataclass(frozen=True)
class DeepFeatureSpec:
    """Feature columns used by a real-data deep QSAR run."""

    descriptor_columns: list[str]
    fingerprint_columns: list[str]
    endpoint_columns: list[str]
    species_context_columns: list[str]
    species_numeric_columns: list[str]
    species_categorical_columns: list[str]
    use_duration: bool


@dataclass(frozen=True)
class DeepTrainingResult:
    """Artifacts returned from a real-data deep QSAR training run."""

    report: dict[str, Any]
    predictions: pd.DataFrame
    model: Any


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for qsar_dl.training.train_deep. "
            "Install torch or the optional project dependency group `deep`."
        ) from _TORCH_IMPORT_ERROR


def build_synthetic_deep_qsar_batch(
    batch_size: int = 16,
    descriptor_group_dim: int = 4,
    descriptor_global_dim: int = 1,
    fingerprint_dim: int = 16,
    species_context_dim: int = 3,
    rule_feature_dim: int = 2,
    seed: int = 20260524,
    device: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic tensor batch that does not depend on real data."""

    _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    device_name = _resolve_device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    descriptor_group = torch.randn(
        (batch_size, descriptor_group_dim), generator=generator, dtype=torch.float32
    )
    descriptor_global = torch.randn(
        (batch_size, descriptor_global_dim), generator=generator, dtype=torch.float32
    )
    fingerprint = torch.randint(
        low=0,
        high=2,
        size=(batch_size, fingerprint_dim),
        generator=generator,
    ).float()
    species_context = torch.randn(
        (batch_size, species_context_dim), generator=generator, dtype=torch.float32
    )
    duration_h = 1.0 + 120.0 * torch.rand(
        (batch_size, 1), generator=generator, dtype=torch.float32
    )
    rule_features = torch.randn(
        (batch_size, rule_feature_dim), generator=generator, dtype=torch.float32
    )

    target = (
        0.45 * descriptor_group[:, :1]
        - 0.20 * descriptor_group[:, 1:2]
        + 0.15 * descriptor_global[:, :1]
        + 0.10 * fingerprint[:, :4].mean(dim=1, keepdim=True)
        + 0.18 * species_context[:, :1]
        + 0.05 * torch.log1p(duration_h)
        + 0.12 * rule_features[:, :1]
    )
    target = target + 0.01 * torch.randn(
        (batch_size, 1), generator=generator, dtype=torch.float32
    )

    batch = {
        "descriptor_group": descriptor_group,
        "descriptor_global": descriptor_global,
        "fingerprint": fingerprint,
        "species_context": species_context,
        "duration_h": duration_h,
        "rule_features": rule_features,
        "target": target,
    }
    return {key: value.to(device_name) for key, value in batch.items()}


def run_small_train_smoke(
    steps: int = 6,
    batch_size: int = 16,
    seed: int = 20260524,
    lr: float = 0.01,
    device: str | None = None,
) -> dict[str, Any]:
    """Run a tiny deterministic optimization loop for integration smoke checks."""

    _require_torch()
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    if lr <= 0:
        raise ValueError("lr must be > 0.")

    device_name = _resolve_device(device)
    torch.manual_seed(int(seed))
    batch = build_synthetic_deep_qsar_batch(
        batch_size=batch_size,
        seed=seed,
        device=device_name,
    )
    model = ResidualQSARModel(
        descriptor_group_dim=batch["descriptor_group"].shape[1],
        descriptor_global_dim=batch["descriptor_global"].shape[1],
        fingerprint_dim=batch["fingerprint"].shape[1],
        species_context_dim=batch["species_context"].shape[1],
        rule_feature_dim=batch["rule_features"].shape[1],
        chemical_hidden_dims=(32,),
        context_hidden_dims=(16,),
        rule_hidden_dims=(16,),
        head_hidden_dims=(16,),
        chemical_embedding_dim=32,
        species_embedding_dim=8,
        time_embedding_dim=4,
        rule_embedding_dim=8,
        dropout=0.0,
        alpha_init=0.2,
        beta_init=0.1,
    ).to(device_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = F.mse_loss(output["y_pred"], batch["target"])
        loss = loss + model.residual_regularization(
            output, context_l2=0.01, rule_l2=0.01
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        output = model(batch)

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "n_steps": steps,
        "device": str(device_name),
        "output_shapes": {
            key: tuple(value.shape)
            for key, value in output.items()
            if torch.is_tensor(value)
        },
    }


train_deep_smoke = run_small_train_smoke


class _DeepQSARDataset(Dataset):  # type: ignore[misc]
    def __init__(self, tensors: Mapping[str, Any]) -> None:
        _require_torch()
        if not tensors:
            raise ValueError("tensors must not be empty.")
        lengths = {
            int(value.shape[0])
            for value in tensors.values()
            if torch.is_tensor(value)
        }
        if len(lengths) != 1:
            raise ValueError("all tensors must have the same row count.")
        self.tensors = dict(tensors)
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {key: value[index] for key, value in self.tensors.items()}


def run_real_data_deep_qsar(
    data: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    descriptor_columns: Sequence[str] | None = None,
    fingerprint_columns: Sequence[str] | None = None,
    species_context_columns: Sequence[str] | None = None,
    target_column: str = "target_ptox",
    split_column: str = "split",
    train_values: Sequence[str] = ("train",),
    validation_values: Sequence[str] = ("validation", "val", "test"),
    device: str | None = None,
) -> DeepTrainingResult:
    """Train a residual QSAR deep baseline on a prepared real-data table.

    The first supported real-data mode is intentionally conservative:
    chemical descriptors and Morgan fingerprints form the main signal, while
    endpoint, duration and species-context branches can be enabled by config.
    Targets are standardized on the training split and inverted for metrics.
    """

    _require_torch()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    cfg = dict(config or {})
    training_cfg = dict(cfg.get("training", {}))
    model_cfg = dict(cfg.get("model", {}))
    random_state = _random_seed(cfg, training_cfg)
    _seed_everything(random_state)

    spec = build_deep_feature_spec(
        data,
        config=cfg,
        descriptor_columns=descriptor_columns,
        fingerprint_columns=fingerprint_columns,
        species_context_columns=species_context_columns,
    )
    train_mask, validation_mask = _split_masks(
        data,
        split_column=split_column,
        train_values=train_values,
        validation_values=validation_values,
    )
    target = pd.to_numeric(data[target_column], errors="coerce")
    valid = target.notna()
    train_mask = train_mask & valid
    validation_mask = validation_mask & valid
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("deep training requires non-empty train and validation splits.")

    prepared = _prepare_deep_arrays(
        data,
        spec=spec,
        train_mask=train_mask,
        target_column=target_column,
    )
    device_name = _resolve_device(device or training_cfg.get("device"))
    train_tensors = _make_tensor_dict(prepared, train_mask, device_name)
    validation_tensors = _make_tensor_dict(prepared, validation_mask, device_name)
    train_dataset = _DeepQSARDataset(train_tensors)
    validation_dataset = _DeepQSARDataset(validation_tensors)

    batch_size = int(training_cfg.get("batch_size", 256))
    max_epochs = int(training_cfg.get("max_epochs", 10))
    learning_rate = float(training_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    patience = int(training_cfg.get("patience", max(3, max_epochs)))
    if batch_size <= 0:
        raise ValueError("training.batch_size must be > 0.")
    if max_epochs <= 0:
        raise ValueError("training.max_epochs must be > 0.")
    if learning_rate <= 0:
        raise ValueError("training.learning_rate must be > 0.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = _build_model_from_spec(spec, model_cfg).to(device_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    context_l2 = float(dict(model_cfg.get("residual", {})).get("context_l2", 0.0))
    rule_l2 = float(dict(model_cfg.get("residual", {})).get("rule_l2", 0.0))

    history: list[dict[str, float]] = []
    best_state = None
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            context_l2=context_l2,
            rule_l2=rule_l2,
        )
        validation_loss = _evaluate_loss(model, validation_loader)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1.0e-7:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_predictions = _predict(model, train_loader, prepared["target_mean"], prepared["target_std"])
    validation_predictions = _predict(
        model,
        validation_loader,
        prepared["target_mean"],
        prepared["target_std"],
    )
    predictions = pd.concat(
        [
            _prediction_frame(data.loc[train_mask], train_predictions, "训练集"),
            _prediction_frame(data.loc[validation_mask], validation_predictions, "验证集"),
        ],
        ignore_index=True,
    )
    validation_metrics = evaluate_regression(
        predictions.loc[predictions["数据集"] == "验证集", "观测pTox"],
        predictions.loc[predictions["数据集"] == "验证集", "预测pTox"],
    )
    train_metrics = evaluate_regression(
        predictions.loc[predictions["数据集"] == "训练集", "观测pTox"],
        predictions.loc[predictions["数据集"] == "训练集", "预测pTox"],
    )
    report = {
        "target_column": target_column,
        "random_state": random_state,
        "device": str(device_name),
        "feature_spec": {
            "descriptor_columns": spec.descriptor_columns,
            "fingerprint_columns": spec.fingerprint_columns,
            "endpoint_columns": spec.endpoint_columns,
            "species_context_columns": spec.species_context_columns,
            "species_numeric_columns": spec.species_numeric_columns,
            "species_categorical_columns": spec.species_categorical_columns,
            "use_duration": spec.use_duration,
        },
        "dataset": {
            "row_count": int(len(data)),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "chemical_count": int(data.loc[train_mask | validation_mask, "chemical_id"].nunique())
            if "chemical_id" in data.columns
            else None,
            "target_mean": float(prepared["target_mean"]),
            "target_std": float(prepared["target_std"]),
        },
        "model": {
            "architecture": model_cfg.get("architecture", "residual_qsar_mlp"),
            "use_species": bool(spec.species_context_columns),
            "use_duration": bool(spec.use_duration),
            "use_rules": False,
            "descriptor_group_dim": len(spec.descriptor_columns),
            "fingerprint_dim": len(spec.fingerprint_columns),
            "endpoint_dim": len(spec.endpoint_columns),
            "species_context_dim": len(spec.species_context_columns),
        },
        "training": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "completed_epochs": len(history),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "history": history,
            "best_validation_loss": best_validation_loss,
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
        },
    }
    return DeepTrainingResult(report=report, predictions=predictions, model=model)


def build_deep_feature_spec(
    data: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    descriptor_columns: Sequence[str] | None = None,
    fingerprint_columns: Sequence[str] | None = None,
    species_context_columns: Sequence[str] | None = None,
) -> DeepFeatureSpec:
    """Resolve real-data deep model feature columns from data and config."""

    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model", {}))
    feature_cfg = dict(cfg.get("deep_features", cfg.get("features", {})))
    descriptor_requested = descriptor_columns or feature_cfg.get("descriptor_columns")
    if descriptor_requested is None:
        descriptor_requested = DEFAULT_DESCRIPTOR_COLUMNS
    descriptor_resolved = [
        str(column)
        for column in descriptor_requested
        if str(column) in data.columns
    ]
    fp_requested = fingerprint_columns or feature_cfg.get("fingerprint_columns")
    if fp_requested is None:
        prefix = str(feature_cfg.get("fingerprint_prefix", DEFAULT_FINGERPRINT_PREFIX))
        fingerprint_resolved = [
            column
            for column in data.columns
            if str(column).startswith(prefix)
        ]
    else:
        fingerprint_resolved = [
            str(column)
            for column in fp_requested
            if str(column) in data.columns
        ]
    context_cfg = dict(model_cfg.get("context_encoder", {}))
    endpoint_columns = (
        _one_hot_columns(data, "endpoint_family")
        if bool(context_cfg.get("use_endpoint", False))
        else []
    )
    requested_species = species_context_columns or feature_cfg.get("species_context_columns")
    species_numeric: list[str] = []
    species_categorical: list[str] = []
    species_resolved: list[str] = []
    for raw_column in requested_species or ():
        column = str(raw_column)
        if column not in data.columns:
            continue
        if is_numeric_dtype(data[column]) or is_bool_dtype(data[column]):
            species_numeric.append(column)
            species_resolved.append(column)
        else:
            species_categorical.append(column)
            species_resolved.extend(_one_hot_columns(data, column))
    return DeepFeatureSpec(
        descriptor_columns=descriptor_resolved,
        fingerprint_columns=fingerprint_resolved,
        endpoint_columns=endpoint_columns,
        species_context_columns=species_resolved,
        species_numeric_columns=species_numeric,
        species_categorical_columns=species_categorical,
        use_duration=bool(context_cfg.get("use_duration", False)),
    )


def _prepare_deep_arrays(
    data: pd.DataFrame,
    *,
    spec: DeepFeatureSpec,
    train_mask: pd.Series,
    target_column: str,
) -> dict[str, Any]:
    if not spec.descriptor_columns and not spec.fingerprint_columns:
        raise ValueError("deep QSAR requires descriptor or fingerprint features.")

    arrays: dict[str, Any] = {}
    descriptor = _numeric_frame(data, spec.descriptor_columns)
    descriptor_scaled, descriptor_center, descriptor_scale = _standardize_frame(
        descriptor,
        train_mask=train_mask,
    )
    arrays["descriptor_group"] = descriptor_scaled
    arrays["descriptor_center"] = descriptor_center
    arrays["descriptor_scale"] = descriptor_scale

    if spec.fingerprint_columns:
        arrays["fingerprint"] = _numeric_frame(data, spec.fingerprint_columns).fillna(0.0)
    else:
        arrays["fingerprint"] = pd.DataFrame(index=data.index)

    if spec.endpoint_columns:
        arrays["endpoint_features"] = _one_hot_frame(data, "endpoint_family", spec.endpoint_columns)
    else:
        arrays["endpoint_features"] = pd.DataFrame(index=data.index)

    if spec.species_context_columns:
        species_parts: list[pd.DataFrame] = []
        if spec.species_numeric_columns:
            species_parts.append(_numeric_frame(data, spec.species_numeric_columns))
        for categorical_column in spec.species_categorical_columns:
            encoded_columns = [
                column
                for column in spec.species_context_columns
                if column.startswith(f"{categorical_column}_")
            ]
            if encoded_columns:
                species_parts.append(_one_hot_frame(data, categorical_column, encoded_columns))
        species = pd.concat(species_parts, axis=1) if species_parts else pd.DataFrame(index=data.index)
        species_scaled, species_center, species_scale = _standardize_frame(
            species,
            train_mask=train_mask,
        )
        arrays["species_context"] = species_scaled
        arrays["species_center"] = species_center
        arrays["species_scale"] = species_scale
    else:
        arrays["species_context"] = pd.DataFrame(index=data.index)

    if spec.use_duration:
        duration = _numeric_frame(data, ["duration_h"]).fillna(0.0)
        arrays["duration_h"] = duration
    else:
        arrays["duration_h"] = pd.DataFrame(index=data.index)

    target = pd.to_numeric(data[target_column], errors="coerce")
    target_train = target.loc[train_mask]
    target_mean = float(target_train.mean())
    target_std = float(target_train.std(ddof=0))
    if not math.isfinite(target_std) or target_std <= 1.0e-12:
        target_std = 1.0
    arrays["target"] = ((target - target_mean) / target_std).to_frame("target")
    arrays["target_raw"] = target.to_frame(target_column)
    arrays["target_mean"] = target_mean
    arrays["target_std"] = target_std
    return arrays


def _make_tensor_dict(
    arrays: Mapping[str, Any],
    mask: pd.Series,
    device: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "descriptor_group": _frame_to_tensor(arrays["descriptor_group"], mask, device),
        "target": _frame_to_tensor(arrays["target"], mask, device),
        "target_raw": _frame_to_tensor(arrays["target_raw"], mask, device),
    }
    if not arrays["fingerprint"].empty:
        output["fingerprint"] = _frame_to_tensor(arrays["fingerprint"], mask, device)
    if not arrays["endpoint_features"].empty:
        output["endpoint_features"] = _frame_to_tensor(arrays["endpoint_features"], mask, device)
    if not arrays["species_context"].empty:
        output["species_context"] = _frame_to_tensor(arrays["species_context"], mask, device)
    if not arrays["duration_h"].empty:
        output["duration_h"] = _frame_to_tensor(arrays["duration_h"], mask, device)
    return output


def _build_model_from_spec(spec: DeepFeatureSpec, model_cfg: Mapping[str, Any]) -> Any:
    chemical_cfg = dict(model_cfg.get("chemical_encoder", {}))
    context_cfg = dict(model_cfg.get("context_encoder", {}))
    residual_cfg = dict(model_cfg.get("residual", {}))
    return ResidualQSARModel(
        descriptor_group_dim=len(spec.descriptor_columns),
        descriptor_global_dim=0,
        fingerprint_dim=len(spec.fingerprint_columns),
        species_context_dim=len(spec.species_context_columns),
        rule_feature_dim=0,
        endpoint_dim=len(spec.endpoint_columns),
        chemical_hidden_dims=tuple(chemical_cfg.get("hidden_dims", (512, 256))),
        context_hidden_dims=tuple(context_cfg.get("hidden_dims", (128,))),
        rule_hidden_dims=(64,),
        head_hidden_dims=tuple(model_cfg.get("head_hidden_dims", (128, 64))),
        chemical_embedding_dim=int(chemical_cfg.get("embedding_dim", 256)),
        species_embedding_dim=int(context_cfg.get("species_embedding_dim", 32)),
        time_embedding_dim=int(context_cfg.get("time_embedding_dim", 16)),
        rule_embedding_dim=1,
        dropout=float(chemical_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        use_species=bool(spec.species_context_columns),
        use_duration=bool(spec.use_duration),
        use_rules=False,
        alpha_init=float(residual_cfg.get("alpha_init", 0.2)),
        beta_init=float(residual_cfg.get("beta_init", 0.0)),
    )


def _train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    *,
    context_l2: float,
    rule_l2: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = F.mse_loss(output["y_pred"], batch["target"])
        loss = loss + model.residual_regularization(
            output,
            context_l2=context_l2,
            rule_l2=rule_l2,
        )
        loss.backward()
        optimizer.step()
        rows = int(batch["target"].shape[0])
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


def _evaluate_loss(model: Any, loader: Any) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            loss = F.mse_loss(output["y_pred"], batch["target"])
            rows = int(batch["target"].shape[0])
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
    return total_loss / max(total_rows, 1)


def _predict(
    model: Any,
    loader: Any,
    target_mean: float,
    target_std: float,
) -> dict[str, np.ndarray]:
    model.eval()
    true_raw: list[np.ndarray] = []
    pred_raw: list[np.ndarray] = []
    chemical_raw: list[np.ndarray] = []
    context_raw: list[np.ndarray] = []
    uncertainty_raw: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            pred = output["y_pred"].detach().cpu().numpy().reshape(-1)
            chemical = output["y_chemical"].detach().cpu().numpy().reshape(-1)
            context = output["y_context_residual"].detach().cpu().numpy().reshape(-1)
            uncertainty = output["uncertainty"]
            true_raw.append(batch["target_raw"].detach().cpu().numpy().reshape(-1))
            pred_raw.append(pred * target_std + target_mean)
            chemical_raw.append(chemical * target_std + target_mean)
            context_raw.append(context * target_std)
            if uncertainty is not None:
                uncertainty_raw.append(
                    uncertainty.detach().cpu().numpy().reshape(-1) * target_std
                )
    return {
        "y_true": np.concatenate(true_raw),
        "y_pred": np.concatenate(pred_raw),
        "y_chemical": np.concatenate(chemical_raw),
        "y_context_residual": np.concatenate(context_raw),
        "uncertainty": np.concatenate(uncertainty_raw)
        if uncertainty_raw
        else np.full(sum(len(values) for values in true_raw), np.nan),
    }


def _prediction_frame(
    metadata: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    dataset_label: str,
) -> pd.DataFrame:
    columns = [
        "record_id",
        "chemical_id",
        "endpoint_family",
        "chemical_class_l2",
        "taxon_group_l2",
    ]
    output = metadata[[column for column in columns if column in metadata.columns]].reset_index(drop=True)
    output["数据集"] = dataset_label
    output["观测pTox"] = predictions["y_true"]
    output["预测pTox"] = predictions["y_pred"]
    output["化学主效应pTox"] = predictions["y_chemical"]
    output["上下文残差pTox"] = predictions["y_context_residual"]
    output["预测不确定度"] = predictions["uncertainty"]
    output["预测残差"] = output["预测pTox"] - output["观测pTox"]
    output["绝对误差"] = output["预测残差"].abs()
    return output


def _resolve_device(device: str | None) -> torch.device:
    _require_torch()
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _split_masks(
    data: pd.DataFrame,
    *,
    split_column: str,
    train_values: Sequence[str],
    validation_values: Sequence[str],
) -> tuple[pd.Series, pd.Series]:
    if split_column not in data.columns:
        raise ValueError(f"split column not found: {split_column}")
    split = data[split_column].astype("string")
    train_mask = split.isin([str(value) for value in train_values])
    validation_mask = split.isin([str(value) for value in validation_values])
    return train_mask, validation_mask


def _numeric_frame(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=data.index)
    return data[list(columns)].apply(pd.to_numeric, errors="coerce").astype("float32")


def _standardize_frame(
    frame: pd.DataFrame,
    *,
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if frame.empty:
        return frame, pd.Series(dtype="float32"), pd.Series(dtype="float32")
    train = frame.loc[train_mask]
    center = train.median(axis=0, skipna=True)
    filled = frame.fillna(center).fillna(0.0)
    scale = train.fillna(center).std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return ((filled - center) / scale).astype("float32"), center, scale


def _frame_to_tensor(frame: pd.DataFrame, mask: pd.Series, device: Any) -> Any:
    array = frame.loc[mask].to_numpy(dtype="float32", copy=True)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _one_hot_columns(data: pd.DataFrame, column: str) -> list[str]:
    if column not in data.columns:
        return []
    values = sorted(str(value) for value in data[column].dropna().astype(str).unique())
    return [f"{column}_{value}" for value in values]


def _one_hot_frame(data: pd.DataFrame, column: str, output_columns: Sequence[str]) -> pd.DataFrame:
    if column not in data.columns or not output_columns:
        return pd.DataFrame(index=data.index)
    encoded = pd.get_dummies(data[column].astype("string").fillna("__missing__"), prefix=column, dtype="float32")
    return encoded.reindex(columns=list(output_columns), fill_value=0.0).astype("float32")


def _random_seed(config: Mapping[str, Any], training_config: Mapping[str, Any]) -> int:
    if "random_state" in training_config:
        return int(training_config["random_state"])
    experiment = config.get("experiment")
    if isinstance(experiment, Mapping) and "seed" in experiment:
        return int(experiment["seed"])
    return 20260524


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
