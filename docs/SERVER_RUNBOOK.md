# Remote Server Runbook

This project supports remote training through PowerShell helper scripts. The
scripts are intentionally dry-run first and do not store passwords, tokens, or
private keys.

## Configuration

Edit:

```text
configs/runtime/gpu_server.yaml
```

Required fields:

```yaml
server:
  host: ""
  user: ""
  project_dir: ""
  conda_env: ""
  cuda_visible_devices: "0"
  ssh_port: 22
training:
  experiment_config: configs/experiments/main_residual_qsar.yaml
  output_dir: outputs/models
```

## Dry Runs

Preview project sync:

```powershell
.\scripts\sync_to_server.ps1 -DryRun
```

Preview remote training:

```powershell
.\scripts\train_remote.ps1 -DryRun
```

Preview output sync:

```powershell
.\scripts\sync_from_server.ps1 -DryRun
```

## Reproducibility

The scripts record the current local Git commit hash in the generated command.
Each real training run must also save `config_resolved.yaml` under the experiment
output directory.
