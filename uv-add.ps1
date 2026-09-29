param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$AddArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$cwd = (Get-Location).Path
$driveRoot = [System.IO.Path]::GetPathRoot($cwd)
if (-not $driveRoot) {
    throw "Could not determine drive root from current directory: $cwd"
}

$cacheDir = Join-Path $driveRoot ".uv-cache"
if (-not (Test-Path -Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir | Out-Null
}

$env:UV_CACHE_DIR = $cacheDir

Write-Host "UV_CACHE_DIR=$env:UV_CACHE_DIR"
Write-Host "Running: uv add $($AddArgs -join ' ')"

& uv add @AddArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    exit $exitCode
}