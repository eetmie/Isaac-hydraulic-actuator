# Resume both frozen queues after a reboot or a transient CUDA failure.
param(
    [string]$RunRoot = (Join-Path $HOME "IsaacTraining/runs/weekend_20261113"),
    [string]$Python = (Join-Path $HOME "IsaacLab/.venv/Scripts/python.exe"),
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$runRootPath = (Resolve-Path -LiteralPath $RunRoot).Path
$quickOut = Join-Path $runRootPath "quick"
$studyOut = Join-Path $runRootPath "study"
$quick = Get-Content -LiteralPath (Join-Path $quickOut "plan.json") -Raw | ConvertFrom-Json
$study = Get-Content -LiteralPath (Join-Path $studyOut "plan.json") -Raw | ConvertFrom-Json
foreach ($required in @($Python, $quick.repository, $quick.base, $study.repository, $study.base)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required training path missing: $required"
    }
}
if ([IO.Path]::GetFullPath($study.quick) -ne [IO.Path]::GetFullPath($quickOut)) {
    throw "Study plan points to a different quick queue: $($study.quick)"
}
Write-Output "Quick queue: $quickOut"
Write-Output "Study queue: $studyOut"
Write-Output "Frozen target: $($study.finish_at)"
Write-Output "Completed stages are skipped; interrupted PPO stages resume their saved checkpoints."
if ($PlanOnly) { return }

$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:PYTHONUNBUFFERED = "1"

# Exercise CUDA before either runner changes its status or launches work.
& $Python -B -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.ones(1024).cuda().sum().item())'
if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed; neither queue was started." }

Push-Location -LiteralPath $quick.repository
try {
    & $Python -u -m training.rotation_ablation --base $quick.base --out $quickOut --iterations $quick.iterations
    if ($LASTEXITCODE -ne 0) { throw "Quick queue failed. Inspect $quickOut/status.json and its named log." }
} finally {
    Pop-Location
}

Push-Location -LiteralPath $study.repository
try {
    & $Python -u -m training.weekend_comparison --base $study.base --quick $quickOut --out $studyOut --finish_at $study.finish_at
    if ($LASTEXITCODE -ne 0) { throw "Study queue failed. Inspect $studyOut/status.json and its named log." }
} finally {
    Pop-Location
}
