param(
    [string]$RuntimeConfig = "configs\runtime\gpu_server.yaml",
    [string]$ExperimentConfig = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Commit = git -C $RepoRoot rev-parse --short HEAD

if ($ExperimentConfig -eq "") {
    $ExperimentConfig = "configs/experiments/main_residual_qsar.yaml"
}

$Command = @(
    "cd <server_project_dir>",
    "git rev-parse --short HEAD",
    "python -m qsar_dl.cli train --config $ExperimentConfig --git-commit $Commit"
) -join " && "

if ($DryRun) {
    Write-Output "DRY RUN: remote training command"
    Write-Output "Runtime config: $RuntimeConfig"
    Write-Output "Experiment config: $ExperimentConfig"
    Write-Output "Git commit: $Commit"
    Write-Output $Command
    exit 0
}

throw "Remote execution is not configured yet. Fill configs/runtime/gpu_server.yaml and use sync_to_server.ps1 first."
