# build.ps1 - Compiles MissionControl.exe using PyInstaller

Write-Host "Building MissionControl executable..."
pyinstaller --name MissionControl `
            --onefile `
            --windowed `
            --clean `
            --log-level WARN `
            --paths src `
            src\crypto_research\runner.py

Write-Host "Build Complete! Check the 'dist' folder for MissionControl.exe"
