param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$entrypoint = Join-Path $PSScriptRoot "train_benchmark.py"
$analysisScript = Join-Path $PSScriptRoot "analyze_organsmnist_no_hflip_sensitivity.py"
$outputRoot = Join-Path $PSScriptRoot "outputs\q2_postconfirmatory_organsmnist_no_hflip"
$originalRoot = Join-Path $PSScriptRoot "outputs\q2_confirmatory"
$workbook = Join-Path $outputRoot "live_results_organsmnist_no_hflip.xlsx"
$log = Join-Path $outputRoot "organsmnist_no_hflip_console.log"
$completionMarker = Join-Path $outputRoot "ORGANSMNIST_NO_HFLIP_COMPLETE.txt"
$mutexName = "Local\LGTS_CIFAR_v2_OrganSMNIST_NoHFlip_Sensitivity"

$models = @("ResNet18", "MobileNetV2", "EfficientNetV2S")
$activations = @("LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native")
$studySeedOrder = @(936137, 529212, 448132, 942529, 110107)
$plannedRuns = $models.Count * $activations.Count * $studySeedOrder.Count

foreach ($required in @($python, $entrypoint, $analysisScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file was not found: $required"
    }
}
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

if (-not $DryRun -and (Test-Path -LiteralPath $completionMarker -PathType Leaf)) {
    Write-Host "DONE | Sensitivity completion marker already exists; no training was started."
    exit 0
}

$commonArgs = @(
    "--lgts_alphas", "0.40", "--lgts_beta", "1.0",
    "--lgts_alpha_min", "0.05", "--lgts_alpha_max", "1.5",
    "--lgts_beta_min", "0.5", "--lgts_beta_max", "3.0",
    "--activation_scope", "full_backbone", "--protocol", "full_train",
    "--seed_order", "936137", "529212", "448132", "942529", "110107",
    "--no_pretrained", "--epochs", "100", "--patience", "0", "--min_epochs", "0", "--min_delta", "0",
    "--batch_size", "128", "--med_batch_size", "128",
    "--num_workers", "4", "--prefetch_factor", "4",
    "--optimizer", "adamw", "--fused_optimizer", "--tf32", "--cudnn_benchmark",
    "--scheduler", "cosine", "--learning_rate", "1e-3", "--min_learning_rate", "1e-6", "--weight_decay", "1e-4",
    "--augmentation", "organsmnist_no_hflip", "--normalization", "dataset",
    "--small_input_stem", "auto", "--class_weight", "auto", "--med_size", "64",
    "--plots", "all", "--device", "cuda", "--best_state_device", "cpu",
    "--data_dir", ".\data", "--output_dir", $outputRoot,
    "--live_workbook", $workbook, "--shared_live_workbook"
)

Write-Host "POST-CONFIRMATORY ORGANSMNIST NO-HFLIP SENSITIVITY"
Write-Host "Planned runs | $plannedRuns"
Write-Host "Main confirmatory panel remains unchanged | $originalRoot"

if ($DryRun) {
    $dryArgs = @("--datasets", "organsmnist", "--models") + $models + @("--activations") + $activations + @("--seeds") + $studySeedOrder + $commonArgs + @("--dry_run")
    & $python $entrypoint @dryArgs
    exit $LASTEXITCODE
}

$mutex = [System.Threading.Mutex]::new($false, $mutexName)
$hasLock = $false
try {
    $hasLock = $mutex.WaitOne(0)
    if (-not $hasLock) {
        Write-Host "ACTIVE | Another protected sensitivity instance is already running."
        exit 0
    }
    foreach ($seed in $studySeedOrder) {
        foreach ($model in $models) {
            foreach ($activation in $activations) {
                $label = "organsmnist | $model | seed=$seed | activation=$activation"
                $stageArgs = @(
                    "--datasets", "organsmnist", "--models", $model,
                    "--activations", $activation, "--seeds", "$seed"
                ) + $commonArgs
                for ($attempt = 1; $attempt -le 3; $attempt++) {
                    Write-Host "RUN | $label | attempt=$attempt/3"
                    $ErrorActionPreference = "Continue"
                    & $python $entrypoint @stageArgs 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log -Append
                    $exitCode = $LASTEXITCODE
                    $ErrorActionPreference = "Stop"
                    if ($exitCode -eq 0) { break }
                    if ($attempt -lt 3) { Start-Sleep -Seconds 10 }
                    else { throw "$label failed after 3 attempts. See $log" }
                }
            }
        }
    }
    & $python $analysisScript --sensitivity-root $outputRoot --original-root $originalRoot --expected-runs $plannedRuns
    if ($LASTEXITCODE -ne 0) { throw "Sensitivity analysis failed with exit code $LASTEXITCODE" }
    @(
        "completed=$(Get-Date -Format o)",
        "stage=postconfirmatory_organsmnist_no_hflip",
        "planned_runs=$plannedRuns",
        "augmentation=organsmnist_no_hflip"
    ) | Set-Content -LiteralPath $completionMarker -Encoding UTF8
}
finally {
    if ($hasLock) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
