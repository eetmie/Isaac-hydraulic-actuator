# Use an existing Isaac Lab environment without moving this repository.
param(
    [switch]$Smoke,
    [switch]$PlanOnly,
    [double]$Hours = 12,
    [double]$ExtraSeedHours = 0,
    [string]$IsaacLabRoot = (Join-Path $HOME "IsaacLab"),
    [string]$RunRoot = "",
    [string]$DatasetRevision = "local",
    [double]$ActuatorStepSeconds = 0.18,
    [double]$PpoIterationSeconds = 1.05
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $IsaacLabRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Isaac Lab Python missing: $python. Set -IsaacLabRoot to your existing checkout."
}
if (-not $RunRoot) {
    $RunRoot = if ($Smoke) { "runs/overnight_smoke" } else { "runs/overnight" }
}
$runnerArgs = @(
    "-u", "-m", "training.overnight", "--out", $RunRoot, "--hours", $Hours,
    "--extra_seed_hours", $ExtraSeedHours, "--dataset_revision", $DatasetRevision,
    "--actuator_step_seconds", $ActuatorStepSeconds, "--ppo_iteration_seconds", $PpoIterationSeconds
)
if ($Smoke) { $runnerArgs += "--smoke" }
if ($PlanOnly) { $runnerArgs += "--plan_only" }
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:PYTHONUNBUFFERED = "1"
Push-Location $repo
try {
    & $python @runnerArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
