param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$Host.UI.RawUI.WindowTitle = "LGTS Q2 CONFIRMATORY - OFFICIAL TEST"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$entrypoint = Join-Path $PSScriptRoot "train_benchmark.py"
$analysisScript = Join-Path $PSScriptRoot "analyze_q2_confirmatory.py"
$outputRoot = Join-Path $PSScriptRoot "outputs\q2_confirmatory"
$workbook = Join-Path $outputRoot "live_results_q2.xlsx"
$log = Join-Path $outputRoot "q2_confirmatory_console.log"
$completionMarker = Join-Path $outputRoot "Q2_CONFIRMATORY_COMPLETE.txt"
$mutexName = "Local\LGTS_CIFAR_v2_Q2_Confirmatory"

$datasets = @("cifar100", "dermamnist", "organsmnist")
$models = @("ResNet18", "MobileNetV2", "EfficientNetV2S")
$activations = @(
    "LGTS-f4-Alpha",
    "LGTS-f2-Slope",
    "ACON-C",
    "GELU",
    "Native"
)
$studySeedOrder = @(936137, 529212, 448132, 942529, 110107)
$plannedRuns = $datasets.Count * $models.Count * $activations.Count * $studySeedOrder.Count

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "GPU Python environment was not found: $python"
}
if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
    throw "Benchmark entrypoint was not found: $entrypoint"
}
if (-not (Test-Path -LiteralPath $analysisScript -PathType Leaf)) {
    throw "Confirmatory analysis script was not found: $analysisScript"
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

if (-not $DryRun -and (Test-Path -LiteralPath $completionMarker -PathType Leaf)) {
    Write-Host "DONE | Q2 completion marker already exists; no training was started."
    exit 0
}

$commonArgs = @(
    "--lgts_alphas", "0.40",
    "--lgts_beta", "1.0",
    "--lgts_alpha_min", "0.05",
    "--lgts_alpha_max", "1.5",
    "--lgts_beta_min", "0.5",
    "--lgts_beta_max", "3.0",
    "--activation_scope", "full_backbone",
    "--protocol", "full_train",
    "--seed_order", "936137", "529212", "448132", "942529", "110107",
    "--no_pretrained",
    "--epochs", "100",
    "--patience", "0",
    "--min_epochs", "0",
    "--min_delta", "0",
    "--batch_size", "128",
    "--med_batch_size", "128",
    "--num_workers", "4",
    "--prefetch_factor", "4",
    "--optimizer", "adamw",
    "--fused_optimizer",
    "--tf32",
    "--cudnn_benchmark",
    "--scheduler", "cosine",
    "--learning_rate", "1e-3",
    "--min_learning_rate", "1e-6",
    "--weight_decay", "1e-4",
    "--augmentation", "standard",
    "--normalization", "dataset",
    "--small_input_stem", "auto",
    "--class_weight", "auto",
    "--med_size", "64",
    "--plots", "all",
    "--device", "cuda",
    "--best_state_device", "cpu",
    "--data_dir", ".\data",
    "--output_dir", $outputRoot,
    "--live_workbook", $workbook,
    "--shared_live_workbook"
)

Write-Host "Q2 CONFIRMATORY | official test protocol"
Write-Host "Datasets | $($datasets -join ' ')"
Write-Host "Models | $($models -join ' ')"
Write-Host "Activations | $($activations -join ' ')"
Write-Host "Seeds | $($studySeedOrder -join ' ')"
Write-Host "Planned runs | $plannedRuns"
Write-Host "Tracker | $workbook"
Write-Host "Resume | completed runs skip; interrupted run restarts at epoch 1"

if ($DryRun) {
    $dryArgs = @(
        "--datasets"
    ) + $datasets + @(
        "--models"
    ) + $models + @(
        "--activations"
    ) + $activations + @(
        "--seeds"
    ) + $studySeedOrder + $commonArgs + @("--dry_run")
    & $python $entrypoint @dryArgs
    exit $LASTEXITCODE
}

$mutex = [System.Threading.Mutex]::new($false, $mutexName)
$hasLock = $false
try {
    $hasLock = $mutex.WaitOne(0)
    if (-not $hasLock) {
        Write-Host "ACTIVE | Another protected Q2 confirmatory instance is already running."
        exit 0
    }

    foreach ($dataset in $datasets) {
        foreach ($seed in $studySeedOrder) {
            foreach ($model in $models) {
                foreach ($activation in $activations) {
                    $label = "$dataset | $model | seed=$seed | activation=$activation"
                    $stageArgs = @(
                        "--datasets", $dataset,
                        "--models", $model,
                        "--activations", $activation,
                        "--seeds", "$seed"
                    ) + $commonArgs

                    $maxAttempts = 3
                    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
                        Write-Host ""
                        Write-Host "RUN | $label | attempt=$attempt/$maxAttempts"
                        $ErrorActionPreference = "Continue"
                        & $python $entrypoint @stageArgs 2>&1 |
                            ForEach-Object { $_.ToString() } |
                            Tee-Object -FilePath $log -Append
                        $exitCode = $LASTEXITCODE
                        $ErrorActionPreference = "Stop"

                        if ($exitCode -eq 0) {
                            break
                        }
                        if ($attempt -lt $maxAttempts) {
                            Write-Warning (
                                "$label failed with exit code $exitCode. " +
                                "Completed conditions are preserved; retrying in a fresh process."
                            )
                            Start-Sleep -Seconds 10
                        }
                        else {
                            throw "$label failed after $maxAttempts attempts. See $log"
                        }
                    }
                }
            }
        }
    }

    & $python $analysisScript --output_dir $outputRoot --expected_runs $plannedRuns
    if ($LASTEXITCODE -ne 0) {
        throw "Q2 integrity/statistical analysis failed with exit code $LASTEXITCODE."
    }

    @(
        "completed=$(Get-Date -Format o)"
        "stage=q2_confirmatory_official_test"
        "datasets=$($datasets -join ' ')"
        "models=$($models -join ' ')"
        "activations=$($activations -join ' ')"
        "seeds=$($studySeedOrder -join ' ')"
        "planned_runs=$plannedRuns"
        "tracker=$workbook"
    ) | Set-Content -LiteralPath $completionMarker -Encoding UTF8

    Write-Host "Q2 CONFIRMATORY COMPLETE | $completionMarker"
}
finally {
    if ($hasLock) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
