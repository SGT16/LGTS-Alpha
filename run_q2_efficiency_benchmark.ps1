$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$output = Join-Path $PSScriptRoot "outputs\q2_confirmatory\efficiency"
New-Item -ItemType Directory -Force -Path $output | Out-Null

$models = @("ResNet18", "MobileNetV2", "EfficientNetV2S")
$activations = @("LGTS-f4-Alpha", "LGTS-f2-Slope", "ACON-C", "GELU", "Native")

& $python benchmark_speed.py `
    --models $models `
    --activations $activations `
    --batch_size 128 `
    --image_size 32 `
    --num_classes 100 `
    --warmup 10 `
    --repeats 30 `
    --modes tf32 `
    --output_csv (Join-Path $output "efficiency_cifar100_32px.csv")
if ($LASTEXITCODE -ne 0) { throw "CIFAR-100 efficiency benchmark failed." }

& $python benchmark_speed.py `
    --models $models `
    --activations $activations `
    --batch_size 128 `
    --image_size 64 `
    --num_classes 11 `
    --warmup 10 `
    --repeats 30 `
    --modes tf32 `
    --output_csv (Join-Path $output "efficiency_medmnist_64px.csv")
if ($LASTEXITCODE -ne 0) { throw "MedMNIST efficiency benchmark failed." }

Write-Host "Q2 EFFICIENCY BENCHMARK COMPLETE | $output"
