"""Small deep-training helpers for residual QSAR smoke tests."""

from __future__ import annotations

from typing import Any

from qsar_dl.models.residual_qsar import ResidualQSARModel

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - depends on optional install.
    torch = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


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


def _resolve_device(device: str | None) -> torch.device:
    _require_torch()
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
