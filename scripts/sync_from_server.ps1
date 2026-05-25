param(
    [string]$RuntimeConfig = "configs\runtime\gpu_server.yaml",
    [string]$RemoteOutput = "outputs/models",
    [string]$LocalOutput = "outputs/models",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Command = "rsync -av <user>@<host>:<server_project_dir>/$RemoteOutput $LocalOutput"

if ($DryRun) {
    Write-Output "DRY RUN: sync outputs from server"
    Write-Output "Runtime config: $RuntimeConfig"
    Write-Output "Remote output: $RemoteOutput"
    Write-Output "Local output: $LocalOutput"
    Write-Output $Command
    exit 0
}

throw "Remote output sync is not configured yet. Fill configs/runtime/gpu_server.yaml with host/user/project_dir."
