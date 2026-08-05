param(
    [string]$Workspace = "D:\trabajo_PLN\Trabajo_PLN-MIA-Grupo4"
)

$ErrorActionPreference = "Stop"
$workspaceRoot = [System.IO.Path]::GetFullPath($Workspace)
$archiveRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $workspaceRoot "Cuadernos\04_old_5etiquetas\artefactos")
)
if (-not $archiveRoot.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "El archivo quedó fuera del workspace."
}

$relativeDirectories = @(
    "resultados\metricas\moderador_grueso",
    "resultados\figuras\moderador_grueso",
    "modelos\moderador_grueso",
    "resultados\metricas\moderador_grueso_mejorado",
    "resultados\figuras\moderador_grueso_mejorado",
    "modelos\moderador_grueso_mejorado",
    "resultados\metricas\moderador_flujo2",
    "resultados\figuras\moderador_flujo2",
    "modelos\moderador_flujo2",
    "resultados\metricas\transformer_grueso",
    "resultados\figuras\transformer_grueso",
    "modelos\moderador_transformer_grueso",
    "resultados\metricas\jerarquico_clasico",
    "resultados\figuras\jerarquico_clasico",
    "modelos\jerarquico_clasico"
)

$relativeReports = @(
    "resultados\CRITERIOS_SELECCION_MODELOS_TRANSFORMER.md",
    "resultados\CRITERIOS_SELECCION_QWEN3_LORA.md",
    "resultados\INFORME_PRIMER_ENTRENAMIENTO_MODELOS_GRUESOS.md",
    "resultados\INFORME_SEGUNDO_ENTRENAMIENTO_MEJORAS.md",
    "resultados\INFORME_ENTRENAMIENTO_TRANSFORMER_GRUESO.md",
    "resultados\INFORME_DECISION_OPERATIVA_MODERADOR.md",
    "resultados\INFORME_EXPERIMENTO_JERARQUICO_CLASICO.md"
)

function Copy-VerifiedFile {
    param([string]$Source, [string]$Destination)
    $sourcePath = [System.IO.Path]::GetFullPath($Source)
    $destinationPath = [System.IO.Path]::GetFullPath($Destination)
    if (
        -not $sourcePath.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $destinationPath.StartsWith($archiveRoot, [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Ruta fuera del alcance: $sourcePath -> $destinationPath"
    }
    $sourceItem = Get-Item -LiteralPath $sourcePath
    $copyNeeded = -not (Test-Path -LiteralPath $destinationPath -PathType Leaf)
    if (-not $copyNeeded) {
        $destinationItem = Get-Item -LiteralPath $destinationPath
        $copyNeeded = $destinationItem.Length -ne $sourceItem.Length
    }
    if ($copyNeeded) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $destinationPath) -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    }
}

foreach ($relativeDirectory in $relativeDirectories) {
    $sourceDirectory = Join-Path $workspaceRoot $relativeDirectory
    if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) {
        continue
    }
    foreach ($sourceFile in Get-ChildItem -LiteralPath $sourceDirectory -Recurse -File) {
        $relativeFile = $sourceFile.FullName.Substring($workspaceRoot.Length).TrimStart("\")
        Copy-VerifiedFile -Source $sourceFile.FullName -Destination (
            Join-Path $archiveRoot $relativeFile
        )
    }
}

foreach ($relativeReport in $relativeReports) {
    $sourceFile = Join-Path $workspaceRoot $relativeReport
    if (Test-Path -LiteralPath $sourceFile -PathType Leaf) {
        Copy-VerifiedFile -Source $sourceFile -Destination (Join-Path $archiveRoot $relativeReport)
    }
}

$sourceFiles = foreach ($relativeDirectory in $relativeDirectories) {
    $sourceDirectory = Join-Path $workspaceRoot $relativeDirectory
    if (Test-Path -LiteralPath $sourceDirectory -PathType Container) {
        Get-ChildItem -LiteralPath $sourceDirectory -Recurse -File
    }
}
$sourceFiles += foreach ($relativeReport in $relativeReports) {
    $sourceFile = Join-Path $workspaceRoot $relativeReport
    if (Test-Path -LiteralPath $sourceFile -PathType Leaf) { Get-Item -LiteralPath $sourceFile }
}

$errors = @()
foreach ($sourceFile in $sourceFiles) {
    $relativeFile = $sourceFile.FullName.Substring($workspaceRoot.Length).TrimStart("\")
    $destinationFile = Join-Path $archiveRoot $relativeFile
    if (-not (Test-Path -LiteralPath $destinationFile -PathType Leaf)) {
        $errors += "ausente:$relativeFile"
    } elseif ((Get-Item -LiteralPath $destinationFile).Length -ne $sourceFile.Length) {
        $errors += "tamano:$relativeFile"
    }
}
if ($errors.Count -gt 0) {
    throw "Archivo incompleto: $($errors -join ', ')"
}

$status = @{
    status = "complete"
    completed_at = [DateTimeOffset]::Now.ToString("o")
    source_files = $sourceFiles.Count
    source_bytes = [int64](($sourceFiles | Measure-Object Length -Sum).Sum)
    note = "Copia archivistica; los originales permanecen para warm-start y 04_7 activo."
}
$status | ConvertTo-Json | Set-Content -LiteralPath (
    Join-Path $archiveRoot "ARCHIVO_COMPLETO.json"
) -Encoding utf8
