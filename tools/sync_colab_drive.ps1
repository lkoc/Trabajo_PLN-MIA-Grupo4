param(
    [Parameter(Mandatory = $true)]
    [string]$DriveRoot
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedDriveRoot = [System.IO.Path]::GetFullPath($DriveRoot)
if ([System.IO.Path]::GetFileName($resolvedDriveRoot) -ne "ModeracionPeru_Colab") {
    throw "DriveRoot debe terminar exactamente en ModeracionPeru_Colab: $resolvedDriveRoot"
}
$bundle = Join-Path $resolvedDriveRoot "bundle"
New-Item -ItemType Directory -Path $bundle -Force | Out-Null

Push-Location $projectRoot
try {
    python tools\prepare_colab_bundle.py --destination $bundle
    if ($LASTEXITCODE -ne 0) {
        throw "Falló la construcción del bundle de Colab"
    }
} finally {
    Pop-Location
}

$required = @(
    "project_core.zip",
    "chunks_v2.jsonl.gz",
    "dataset_5_salidas.jsonl.gz",
    "bundle_manifest.json"
)
foreach ($name in $required) {
    $path = Join-Path $bundle $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Falta archivo después de sincronizar: $path"
    }
}
Write-Output "Bundle mínimo sincronizado y manifestado en $bundle"
