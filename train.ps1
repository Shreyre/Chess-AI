param([int]$Iterations = 100)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $trainingArgs = @('-m', 'chess_ai', 'train', '--iterations', "$Iterations")
    if (Test-Path -LiteralPath 'runs/main/latest.pt') {
        $trainingArgs += @('--resume', 'runs/main/latest.pt')
    }
    & "$PSScriptRoot/.venv/Scripts/python.exe" @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
