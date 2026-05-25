from __future__ import annotations

import math

import pytest

from qsar_dl.models import residual_qsar


def test_residual_qsar_module_import_is_torch_optional() -> None:
    assert hasattr(residual_qsar, "ResidualQSARModel")
    assert hasattr(residual_qsar, "ChemicalEncoder")


torch = pytest.importorskip("torch")

from qsar_dl.models.residual_qsar import ResidualQSARModel  # noqa: E402
from qsar_dl.training.train_deep import run_small_train_smoke  # noqa: E402


def _batch(
    batch_size: int = 5,
    descriptor_group_dim: int = 4,
    descriptor_global_dim: int = 1,
    fingerprint_dim: int = 8,
    species_context_dim: int = 3,
    rule_feature_dim: int = 2,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260524)
    return {
        "descriptor_group": torch.randn(
            (batch_size, descriptor_group_dim), generator=generator
        ),
        "descriptor_global": torch.randn(
            (batch_size, descriptor_global_dim), generator=generator
        ),
        "fingerprint": torch.randint(
            0, 2, (batch_size, fingerprint_dim), generator=generator
        ).float(),
        "species_context": torch.randn(
            (batch_size, species_context_dim), generator=generator
        ),
        "duration_h": 1.0
        + 96.0 * torch.rand((batch_size, 1), generator=generator),
        "rule_features": torch.randn((batch_size, rule_feature_dim), generator=generator),
    }


def _model(**overrides) -> ResidualQSARModel:
    params = {
        "descriptor_group_dim": 4,
        "descriptor_global_dim": 1,
        "fingerprint_dim": 8,
        "species_context_dim": 3,
        "rule_feature_dim": 2,
        "chemical_hidden_dims": (16,),
        "context_hidden_dims": (8,),
        "rule_hidden_dims": (8,),
        "head_hidden_dims": (8,),
        "chemical_embedding_dim": 16,
        "species_embedding_dim": 6,
        "time_embedding_dim": 4,
        "rule_embedding_dim": 5,
        "dropout": 0.0,
        "alpha_init": 0.2,
        "beta_init": 0.1,
    }
    params.update(overrides)
    return ResidualQSARModel(**params)


def test_forward_output_shapes() -> None:
    model = _model()
    batch = _batch()

    output = model(batch)

    assert output["y_pred"].shape == (5, 1)
    assert output["y_chemical"].shape == (5, 1)
    assert output["y_context_residual"].shape == (5, 1)
    assert output["y_rule_residual"].shape == (5, 1)
    assert output["uncertainty"].shape == (5, 1)
    assert torch.all(output["uncertainty"] > 0)
    assert isinstance(output["aux"], dict)
    assert output["aux"]["z_chem"].shape == (5, 16)


def test_can_disable_species_duration_and_rules() -> None:
    model = _model(
        species_context_dim=0,
        rule_feature_dim=0,
        use_species=False,
        use_duration=False,
        use_rules=False,
    )
    batch = _batch()
    batch.pop("species_context")
    batch.pop("duration_h")
    batch.pop("rule_features")

    output = model(batch)

    assert torch.allclose(output["y_context_residual"], torch.zeros_like(output["y_pred"]))
    assert torch.allclose(output["y_rule_residual"], torch.zeros_like(output["y_pred"]))
    assert torch.allclose(output["y_pred"], output["y_chemical"])


def test_residual_decomposition_is_directly_checkable() -> None:
    model = _model()
    output = model(_batch())

    decomposed = (
        output["y_chemical"]
        + output["y_context_residual"]
        + output["y_rule_residual"]
    )

    assert torch.allclose(output["y_pred"], decomposed, atol=1.0e-6)


def test_small_train_smoke_helper_runs_without_real_data() -> None:
    result = run_small_train_smoke(steps=3, batch_size=8, seed=20260524, device="cpu")

    assert result["n_steps"] == 3
    assert len(result["losses"]) == 3
    assert math.isfinite(result["initial_loss"])
    assert math.isfinite(result["final_loss"])
    assert result["output_shapes"]["y_pred"] == (8, 1)
