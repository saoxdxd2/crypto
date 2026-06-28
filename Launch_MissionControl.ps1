$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "      MISSION CONTROL: LAUNCH SEQUENCE    " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check ONNXRuntime
$extractPath = "C:\onnxruntime"
if (-Not (Test-Path "$extractPath\include\onnxruntime_cxx_api.h")) {
    Write-Host "[INIT] ONNXRuntime not found. Automating download..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
} else {
    Write-Host "[INIT] ONNXRuntime validated." -ForegroundColor Green
}

# 2. Compile Zig/C++ Engine
Write-Host "[BUILD] Checking Native Execution Engine..." -ForegroundColor Cyan
Set-Location src\execution
cmake -S . -B build -DONNXRUNTIME_ROOT="C:\onnxruntime" | Out-Null
cmake --build build --config Release | Out-Null
zig build -Doptimize=ReleaseFast | Out-Null
Set-Location ..\..
Write-Host "[BUILD] Native Engine Ready." -ForegroundColor Green

# 3. Compile Go Orchestrator
Write-Host "[BUILD] Checking Go Orchestrator..." -ForegroundColor Cyan
Set-Location src\go_orchestrator
go build -o orchestrator.exe main.go
Set-Location ..\..
Write-Host "[BUILD] Go Orchestrator Ready." -ForegroundColor Green

# 4. Launch Python UI
Write-Host "[LAUNCH] Starting Python Mission Control GUI..." -ForegroundColor Magenta
python src\mission_control\runner.py
