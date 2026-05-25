from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_remote_scripts_are_dry_run_first() -> None:
    for script_name in ("train_remote.ps1", "sync_to_server.ps1", "sync_from_server.ps1"):
        text = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "DryRun" in text
        assert "throw" in text


def test_gpu_server_config_has_no_secret_values() -> None:
    text = (PROJECT_ROOT / "configs" / "runtime" / "gpu_server.yaml").read_text(
        encoding="utf-8"
    )
    assert "password" not in text.lower()
    assert "token" not in text.lower()
    assert "private" not in text.lower()
    assert "project_dir" in text
