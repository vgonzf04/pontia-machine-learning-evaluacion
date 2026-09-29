# This script is used to run uv with a cache directory on the same drive as the current working directory,
# so it handles hardlinks instead of copying on local environment path
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$UvArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Check if help was requested
if ($UvArgs -contains "--help" -or $UvArgs -contains "-h") {
    Write-Host "Usage of uv-sync.ps1 script:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This script runs 'uv' commands while automatically configuring the cache"
    Write-Host "directory on the current drive to optimize hardlinks."
    Write-Host ""
    Write-Host "Available Behaviors:"
    Write-Host "  1. Default behavior (no arguments):"
    Write-Host "     Example: .\uv-sync.ps1"
    Write-Host "     Equivalent to: uv sync (matches V0 behavior)"
    Write-Host ""
    Write-Host "  2. Sync with a requirements file (if you pass an existing file):"
    Write-Host "     Example: .\uv-sync.ps1 requirements.txt"
    Write-Host "     Equivalent to: uv pip sync requirements.txt (cleans and installs exact match)"
    Write-Host ""
    Write-Host "  3. Custom uv commands:"
    Write-Host "     Example: .\uv-sync.ps1 pip install -r requirements.txt"
    Write-Host ""
    exit 0
}

# If no arguments are passed, behave exactly like V0 (@("sync"))
if (-not $UvArgs -or $UvArgs.Count -eq 0) {
    $UvArgs = @("sync")
} else {
    # If the first argument is an existing file (and not a uv option), assume 'uv pip sync'
    $firstArg = $UvArgs[0]
    if (-not $firstArg.StartsWith("-") -and (Test-Path $firstArg)) {
        $UvArgs = @("pip", "sync") + $UvArgs
    }
}

# Use the drive where the script is launched (current working directory).
$cwd = (Get-Location).Path
$driveRoot = [System.IO.Path]::GetPathRoot($cwd)
if (-not $driveRoot) {
    throw "Could not determine drive root from current directory: $cwd"
}

# Keep uv cache on the same drive to allow hardlink mode when possible.
$cacheDir = Join-Path $driveRoot ".uv-cache"
if (-not (Test-Path -Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir | Out-Null
}

$env:UV_CACHE_DIR = $cacheDir

Write-Host "UV_CACHE_DIR=$env:UV_CACHE_DIR"
Write-Host "Running: uv $($UvArgs -join ' ')"

& uv @UvArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    exit $exitCode
}