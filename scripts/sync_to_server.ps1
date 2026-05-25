param(
    [string]$RuntimeConfig = "configs\runtime\gpu_server.yaml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Commit = git -C $RepoRoot rev-parse --short HEAD

$Command = "rsync -av --exclude outputs --exclude data/raw --exclude .git $RepoRoot <user>@<host>:<server_project_dir>"

if ($DryRun) {
    Write-Output "DRY RUN: sync project to server"
    Write-Output "Runtime config: $RuntimeConfig"
    Write-Output "Git commit: $Commit"
    Write-Output $Command
    exit 0
}

throw "Remote sync is not configured yet. Fill configs/runtime/gpu_server.yaml with host/user/project_dir."
