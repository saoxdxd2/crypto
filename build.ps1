# build.ps1 - Compiles MissionControl using PyInstaller
# Usage:
#   powershell -ExecutionPolicy Bypass -File build.ps1              # incremental (fast)
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Clean       # full rebuild (slow)

param(
    [switch]$Clean
)

$cleanFlag = ""
if ($Clean) {
    $cleanFlag = "--clean"
    Write-Host "Full rebuild requested (--clean)..."
} else {
    Write-Host "Incremental build (cached)..."
}

Write-Host "Building MissionControl..."

# Define Native Assets to Bundle
$onnxDll = "C:\onnxruntime\lib\onnxruntime.dll"
$onnxSharedDll = "C:\onnxruntime\lib\onnxruntime_providers_shared.dll"
$zigEngine = "src\execution\zig-out\bin\engine.exe"
$goOrch = "src\go_orchestrator\orchestrator.exe"
$proxy = "src\data\tls_proxy.py"

$args = @(
    "--name", "MissionControl",
    "-y",
    "--onedir",
    "--windowed",
    "--log-level", "WARN",
    "--paths", ".",
    "--exclude-module", "torch._inductor",
    "--exclude-module", "torch.distributed",
    "--exclude-module", "tensorboard",
    "--add-data", "$proxy;src/data",
    "--add-data", "onnx_exports;onnx_exports"
)

if (Test-Path $zigEngine) {
    $args += "--add-binary"
    $args += "$zigEngine;src/execution/zig-out/bin"
} else {
    Write-Host "WARNING: $zigEngine not found. Skipping bundle..." -ForegroundColor Yellow
}

if (Test-Path $goOrch) {
    $args += "--add-binary"
    $args += "$goOrch;src/go_orchestrator"
} else {
    Write-Host "WARNING: $goOrch not found. Skipping bundle..." -ForegroundColor Yellow
}

if (Test-Path $onnxDll) {
    $args += "--add-binary"
    $args += "$onnxDll;src/execution/zig-out/bin"
    $args += "--add-binary"
    $args += "$onnxSharedDll;src/execution/zig-out/bin"
} else {
    Write-Host "WARNING: ONNXRuntime DLLs not found at $onnxDll. Skipping bundle..." -ForegroundColor Yellow
}

if ($Clean) {
    $args += "--clean"
}

$args += "src\mission_control\runner.py"

& pyinstaller @args

Write-Host "Build Complete! Check the 'dist\MissionControl' folder."
