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
$bundle = Join-Path $projectRoot "resultados\colab_bundle"

Push-Location $projectRoot
try {
    python tools\prepare_colab_bundle.py --destination $bundle --drive-root $resolvedDriveRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Falló la construcción del bundle de Colab"
    }
} finally {
    Pop-Location
}

$manifestPath = Join-Path $bundle "bundle_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Falta el manifiesto local después de preparar el bundle: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$release = Join-Path (Join-Path $resolvedDriveRoot "bundle_releases") $manifest.bundle_id
if (-not (Test-Path -LiteralPath (Join-Path $release "bundle_manifest.json") -PathType Leaf)) {
    throw "Falta la versión publicada en Drive: $release"
}
Write-Output "Bundle inmutable publicado en $release"
Write-Output "Espere a que Google Drive termine de sincronizar antes de iniciar Colab."
