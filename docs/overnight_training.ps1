# Overnight actuator + controller training (Windows PowerShell). See docs/overnight_training.md.
# Run from scripts/Isaac-hydraulic-actuator:  powershell -ExecutionPolicy Bypass -File docs/overnight_training.ps1
# Restartable: a finished phase (its selection.json exists) is skipped.
# -Smoke runs the whole chain with tiny budgets (a few minutes) into runs/overnight_smoke; do that first.
param([switch]$Smoke)

$ErrorActionPreference = "Continue"
$R = if ($Smoke) { "runs/overnight_smoke" } else { "runs/overnight" }
# Steps: pretrain, phase A, phases B/C, eval interval, eval cases, compare cases, PPO iterations, PPO envs.
$N = if ($Smoke) { @(200, 20, 20, 10, 8, 8, 2, 256) } else { @(20000, 30000, 20000, 1000, 96, 200, 1500, 4096) }
$pre, $stepsA, $stepsBC, $every, $cases, $cmpCases, $ppoIters, $ppoEnvs = $N
$tag = if ($Smoke) { "smoke_" } else { "" }
$A = "$R/a_scratch_lr1e-4"
$B = "$R/b_lr3e-5"
$C = "$R/c_lr1e-5"
New-Item -ItemType Directory -Force $R | Out-Null

function Step($name, $log, [scriptblock]$cmd) {
    Write-Host "$(Get-Date -Format HH:mm) $name  (log: $log)"
    & $cmd *> $log
    if ($LASTEXITCODE) { Write-Host "FAILED: $name, see $log"; exit 1 }
}

function Py { uv run --project ../.. python @args }

Step "GPU check" "$R/gpu.log" { Py -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" }
Get-Content "$R/gpu.log"

# Same split as the desktop run: Oct 22 robot sessions are the held-out test.
if (-not (Test-Path "$A/split_manifest.json")) {
    Step "prepare split" "$R/prepare.log" {
        Py -m training.prepare --out $A --dev-groups v2_20260909 robot:1792605533932081-71723 `
            --test-groups robot:1792664731108427-pid-3968 robot:1792665379525709-5530 `
            robot:1792668759594343-14974 robot:1792676310556861-26765
    }
}

# Phase A: from scratch, 30k rollout steps. Phases B and C continue from the previous phase's last
# weights with a smaller learning rate.
if (-not (Test-Path "$A/selection.json")) {
    Step "A: scratch $stepsA steps, lr 1e-4" "$A/train.log" {
        Py -m training.actuator train --out $A --scratch --pretrain_steps $pre --steps $stepsA --lr 1e-4 `
            --batch 256 --eval_every $every --cases $cases
    }
}
foreach ($phase in @(@($B, $A, "3e-5"), @($C, $B, "1e-5"))) {
    $out, $parent, $lr = $phase
    if (Test-Path "$out/selection.json") { continue }
    New-Item -ItemType Directory -Force $out | Out-Null
    if (-not (Test-Path "$out/data")) {
        Copy-Item -Recurse "$A/data" "$out/data"
        Copy-Item "$A/split_manifest.json" $out
    }
    Step "$out : $stepsBC steps, lr $lr" "$out/train.log" {
        Py -m training.actuator train --out $out --model "$parent/model/latest" --steps $stepsBC --lr $lr `
            --batch 256 --eval_every $every --cases $cases
    }
}

# Closed-valve bound on each phase's best checkpoint, pick the best on dev, report on test.
foreach ($run in @($A, $B, $C)) {
    if (-not (Test-Path "$run/model_steady")) {
        Step "steady $run" "$run/steady.log" { Py -m training.neutral_bound "$run/model" "$run/model_steady" }
    }
}
$models = @("deployed=models/actuator", "a=$A/model_steady", "b=$B/model_steady", "c=$C/model_steady")
Step "compare on dev" "$R/compare_dev.log" {
    Py -m training.compare_models $A @models --split dev --cases $cmpCases --best_out "$R/best_actuator.txt"
}
Step "compare on test" "$R/compare_test.log" { Py -m training.compare_models $A @models --split test --cases $cmpCases }
Get-Content "$R/compare_dev.log", "$R/compare_test.log"

# PPO pair on two plants (best new actuator + deployed V5): original recipe vs lookahead + acceleration reward.
$best = (Get-Content "$R/best_actuator.txt").Trim() -replace '\\', '/'
$v5 = (Resolve-Path "models/actuator").Path -replace '\\', '/'
$logs = (Resolve-Path $R).Path
Push-Location ../..
foreach ($run in @(@("overnight_base", ""), @("overnight_lookahead_accel", "lookahead_steps=4 accel_weight=0.4"))) {
    $name, $extra = $run
    $name = "$tag$name"
    $sets = @("extra_model_paths=[$v5]") + ($extra -split " " | Where-Object { $_ })
    Step "PPO $name" "$logs/$name.log" {
        uv run python scripts/Isaac-hydraulic-actuator/train_controller.py --run_name $name --model $best `
            --max_iterations $ppoIters --num_envs $ppoEnvs --set @sets
    }
}
Pop-Location
Write-Host "$(Get-Date -Format HH:mm) done. Bring back runs/overnight and logs/rsl_rl/hydraulic_controller/*overnight*"
