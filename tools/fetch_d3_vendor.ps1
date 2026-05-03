param(
    [string]$OutDir = "docs/vendor/d3"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root $OutDir
New-Item -ItemType Directory -Path $target -Force | Out-Null

$files = @(
    @{ Name = "d3.v7.min.js"; Url = "https://d3js.org/d3.v7.min.js" },
    @{ Name = "d3.v7.js"; Url = "https://d3js.org/d3.v7.js" },
    @{ Name = "d3-scale-chromatic.v3.min.js"; Url = "https://d3js.org/d3-scale-chromatic.v3.min.js" }
)

foreach ($item in $files) {
    $dst = Join-Path $target $item.Name
    Write-Host "[FETCH] $($item.Url)"
    Invoke-WebRequest -Uri $item.Url -OutFile $dst
}

Write-Host "[DONE] D3 vendor files downloaded to $target"
