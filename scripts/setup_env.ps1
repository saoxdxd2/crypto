$ErrorActionPreference = "Stop"

$onnxVersion = "1.20.1"
$onnxUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$onnxVersion/onnxruntime-win-x64-$onnxVersion.zip"
$zipPath = "$env:TEMP\onnxruntime.zip"
$extractPath = "C:\onnxruntime"

if (Test-Path "$extractPath\include\onnxruntime_cxx_api.h") {
    Write-Host "ONNXRuntime C++ is already installed at $extractPath. Skipping download." -ForegroundColor Green
} else {
    Write-Host "Downloading ONNXRuntime C++ v$onnxVersion..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $onnxUrl -OutFile $zipPath

    Write-Host "Extracting to $extractPath..." -ForegroundColor Cyan
    if (Test-Path $extractPath) { Remove-Item -Recurse -Force $extractPath }
    Expand-Archive -Path $zipPath -DestinationPath "C:\" -Force

    # Rename extracted folder to standard C:\onnxruntime
    Rename-Item -Path "C:\onnxruntime-win-x64-$onnxVersion" -NewName "onnxruntime"
    
    Remove-Item $zipPath -Force
    Write-Host "ONNXRuntime installed successfully!" -ForegroundColor Green
}

# Compile the Engine
Write-Host "Compiling C++ Inference Core..." -ForegroundColor Cyan
cd src\execution
cmake -S . -B build -DONNXRUNTIME_ROOT="C:\onnxruntime"
cmake --build build --config Release

Write-Host "Compiling Zig Execution Engine..." -ForegroundColor Cyan
zig build -Doptimize=ReleaseFast

Write-Host "Setup Complete!" -ForegroundColor Green
