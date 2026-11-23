<#
.SYNOPSIS
Runs the Go generator and the CUDA generator on the same images, settings and
seed, then prints time and error (RMSE) side by side.

.EXAMPLE
.\bench.ps1 -Images C:\pics\car.png, C:\pics\logo.png -StopAt 500

.NOTES
Both generators are scored by the same code (`--score` in the CUDA EXE), which
paints each geometry file with its exported 8-bit colours, as the game does.
Lower RMSE is better.
#>
param(
    [Parameter(Mandatory = $true)][string[]]$Images,
    [string]$Settings = "",
    [string]$GoExe = "",
    [string]$CudaExe = "",
    [string]$GoBackend = "",
    [int]$Seed = 1,
    [int]$StopAt = 0,
    [string]$OutDir = ".\bench-out"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-First([string[]]$candidates) {
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path } }
    return $null
}

$CudaExe = Find-First @($CudaExe, "$here\forza-painter-geometrize-cuda.exe",
    "$here\build\forza-painter-geometrize-cuda.exe", "$here\..\bin\forza-painter-geometrize-cuda.exe")
$GoExe = Find-First @($GoExe, "$here\forza-painter-geometrize-go.exe", "$here\..\bin\forza-painter-geometrize-go.exe")
$Settings = Find-First @($Settings, "$here\..\config\settings\c. balanced - good quality and speed.ini",
    "$here\c. balanced - good quality and speed.ini")
if (-not $CudaExe) { throw "Can't find forza-painter-geometrize-cuda.exe; pass -CudaExe" }
if (-not $GoExe) { throw "Can't find forza-painter-geometrize-go.exe; pass -GoExe" }
if (-not $Settings) { throw "Can't find the settings file; pass -Settings" }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

# Optionally cap the layer count (and save at the end only).
$ini = $Settings
if ($StopAt -gt 0) {
    $ini = Join-Path $OutDir "bench-settings.ini"
    $lines = Get-Content -LiteralPath $Settings | Where-Object { $_ -notmatch '^\s*(stopAt|saveAt)\s*=' }
    $lines += "stopAt = $StopAt"
    $lines += "saveAt = $StopAt"
    Set-Content -LiteralPath $ini -Value $lines -Encoding ascii
}

Write-Host "Settings: $ini"
Write-Host "Go:       $GoExe"
Write-Host "CUDA:     $CudaExe"
Write-Host ""

$rows = @()
foreach ($img in $Images) {
    $img = (Resolve-Path $img).Path
    $name = [IO.Path]::GetFileNameWithoutExtension($img)
    foreach ($gen in @("go", "cuda")) {
        $exe = if ($gen -eq "go") { $GoExe } else { $CudaExe }
        $dir = Join-Path $OutDir "$name\$gen"
        New-Item -ItemType Directory -Force $dir | Out-Null
        $base = Join-Path $dir "$name.png"
        $runArgs = @($img, "-settings", $ini, "-output", $base, "-seed", "$Seed")
        if ($gen -eq "go" -and $GoBackend) { $runArgs += @("-backend", $GoBackend) }
        Write-Host "Running $gen on $name ..."
        $sw = [Diagnostics.Stopwatch]::StartNew()
        & $exe @runArgs *> (Join-Path $dir "log.txt")
        $code = $LASTEXITCODE
        $sw.Stop()
        if ($code -ne 0) {
            Write-Warning "$gen failed on $name (exit $code); see $(Join-Path $dir 'log.txt')"
            continue
        }
        $json = Join-Path $dir "$name.json"
        $score = & $CudaExe --score $json $img
        $rmse = [double]::NaN; $layers = 0
        foreach ($line in $score) {
            if ($line -match 'RMSE=([0-9.]+)') { $rmse = [double]$Matches[1] }
            if ($line -match 'shapes=(\d+)') { $layers = [int]$Matches[1] }
        }
        $secs = $sw.Elapsed.TotalSeconds
        $rows += [pscustomobject]@{
            Image = $name; Generator = $gen; Layers = $layers
            Seconds = [math]::Round($secs, 1)
            "ms/layer" = if ($layers -gt 0) { [math]::Round(1000 * $secs / $layers, 1) } else { 0 }
            RMSE = [math]::Round($rmse, 3)
        }
    }
}

Write-Host ""
$rows | Format-Table -AutoSize
foreach ($name in ($rows | Select-Object -ExpandProperty Image -Unique)) {
    $go = $rows | Where-Object { $_.Image -eq $name -and $_.Generator -eq "go" }
    $cu = $rows | Where-Object { $_.Image -eq $name -and $_.Generator -eq "cuda" }
    if ($go -and $cu -and $cu.Seconds -gt 0) {
        $speed = [math]::Round($go.Seconds / $cu.Seconds, 1)
        $better = if ($cu.RMSE -le $go.RMSE) { "lower (better)" } else { "higher (worse)" }
        Write-Host "${name}: CUDA is ${speed}x the speed of Go; its RMSE is $better ($($cu.RMSE) vs $($go.RMSE))"
    }
}
Write-Host "Outputs and logs: $OutDir"
