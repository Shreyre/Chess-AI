$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot/.venv/Scripts/python.exe" -m chess_ai @args
    if ($LASTEXITCODE -ne 0) { throw "Chess AI exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
