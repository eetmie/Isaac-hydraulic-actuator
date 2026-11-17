# Use Isaac Lab's uv project without moving this repository.
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
Get-Command uv -ErrorAction Stop | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $IsaacLabRoot "pyproject.toml"))) {
    throw "Isaac Lab project missing: $IsaacLabRoot. Set -IsaacLabRoot to your checkout."
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
    & uv run --project $IsaacLabRoot python @runnerArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
