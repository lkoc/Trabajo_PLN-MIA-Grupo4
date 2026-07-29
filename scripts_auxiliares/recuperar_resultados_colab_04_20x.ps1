param(
    [string]$Workspace = "D:\trabajo_PLN\Trabajo_PLN-MIA-Grupo4",
    [string]$DriveBundle = "G:\My Drive\PLN_colab_04_artifacts",
    [switch]$IncludeQwen,
    [switch]$Qwen04_205Only,
    [switch]$Qwen04_206Only,
    [switch]$Qwen04_20XOnly,
    [switch]$DeploymentOnly,
    [switch]$ComparisonOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$workspaceRoot = [System.IO.Path]::GetFullPath($Workspace)
$driveRoot = [System.IO.Path]::GetFullPath($DriveBundle)
if (-not (Test-Path -LiteralPath $workspaceRoot -PathType Container)) {
    throw "No existe el workspace: $workspaceRoot"
}
if (-not (Test-Path -LiteralPath $driveRoot -PathType Container)) {
    throw "No existe el bundle de Drive: $driveRoot"
}

$relativeDirectories = if ($DeploymentOnly) {
    @(
        "modelos\transformer_plano_4\e5_small",
        "resultados\metricas\transformer_plano_4",
        "modelos\qwen3_06b_lora_acoso_amenaza_4",
        "resultados\metricas\qwen3_06b_lora_acoso_amenaza_4"
    )
} elseif ($Qwen04_20XOnly) {
    @(
        "modelos\qwen3_06b_lora_acoso_amenaza_4",
        "modelos\qwen_jerarquico_4",
        "resultados\metricas\qwen3_06b_lora_acoso_amenaza_4",
        "resultados\metricas\qwen_jerarquico_4",
        "resultados\figuras\qwen3_06b_lora_acoso_amenaza_4",
        "resultados\figuras\qwen_jerarquico_4"
    )
} elseif ($Qwen04_205Only) {
    @(
        "modelos\qwen3_06b_lora_acoso_amenaza_4",
        "resultados\metricas\qwen3_06b_lora_acoso_amenaza_4",
        "resultados\figuras\qwen3_06b_lora_acoso_amenaza_4"
    )
} elseif ($Qwen04_206Only) {
    @(
        "modelos\qwen_jerarquico_4",
        "resultados\metricas\qwen_jerarquico_4",
        "resultados\figuras\qwen_jerarquico_4"
    )
} elseif ($ComparisonOnly) {
    @(
        "resultados\metricas\transformer_plano_4",
        "resultados\metricas\experimentos_jerarquicos_4"
    )
} else {
    @(
        "modelos\transformer_plano_4",
        "modelos\experimentos_jerarquicos_4",
        "resultados\metricas\transformer_plano_4",
        "resultados\metricas\experimentos_jerarquicos_4",
        "resultados\figuras\transformer_plano_4",
        "resultados\figuras\experimentos_jerarquicos_4"
    )
}
$relativeReports = if ($DeploymentOnly) {
    @(
        "resultados\INFORME_TRANSFORMERS_PLANOS_4.md",
        "resultados\INFORME_QWEN_ACOSO_AMENAZA_4.md"
    )
} elseif ($Qwen04_20XOnly) {
    @(
        "resultados\INFORME_QWEN_ACOSO_AMENAZA_4.md",
        "resultados\INFORME_QWEN_JERARQUICO_4.md"
    )
} elseif ($Qwen04_205Only) {
    @(
        "resultados\INFORME_QWEN_ACOSO_AMENAZA_4.md"
    )
} elseif ($Qwen04_206Only) {
    @(
        "resultados\INFORME_QWEN_JERARQUICO_4.md"
    )
} elseif ($ComparisonOnly) {
    @()
} else {
    @(
        "resultados\INFORME_TRANSFORMERS_PLANOS_4.md",
        "resultados\INFORME_04_203_CASCADA_4_ETIQUETAS.md",
        "resultados\INFORME_04_204_JERARQUICO_MULTITAREA_4_ETIQUETAS.md"
    )
}
if ($IncludeQwen -and -not ($Qwen04_205Only -or $Qwen04_206Only -or $Qwen04_20XOnly -or $DeploymentOnly)) {
    if ($ComparisonOnly) {
        $relativeDirectories += @(
            "resultados\metricas\qwen_jerarquico_4"
        )
    } else {
        $relativeDirectories += @(
            "modelos\qwen3_06b_lora_acoso_amenaza_4",
            "modelos\qwen_jerarquico_4",
            "resultados\metricas\qwen3_06b_lora_acoso_amenaza_4",
            "resultados\metricas\qwen_jerarquico_4",
            "resultados\figuras\qwen3_06b_lora_acoso_amenaza_4",
            "resultados\figuras\qwen_jerarquico_4"
        )
        $relativeReports += @(
            "resultados\INFORME_QWEN_ACOSO_AMENAZA_4.md",
            "resultados\INFORME_QWEN_JERARQUICO_4.md"
        )
    }
}

function Copy-BackVerifiedFile {
    param([string]$Source, [string]$RelativePath)
    $destination = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $RelativePath))
    if (-not $destination.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Destino fuera del workspace: $destination"
    }
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.ToLower()
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLower()
        if ($destinationHash -eq $sourceHash) { return $null }
        if (-not $Force) {
            throw "Conflicto local en $RelativePath. Revise el archivo o repita con -Force."
        }
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $destination -Force
    $copiedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLower()
    if ($copiedHash -ne $sourceHash) { throw "Falló la verificación: $RelativePath" }
    return [ordered]@{ path = $RelativePath.Replace("\", "/"); sha256 = $copiedHash; bytes = (Get-Item $destination).Length }
}

$records = @()
foreach ($relativeDirectory in $relativeDirectories) {
    $sourceDirectory = Join-Path $driveRoot $relativeDirectory
    if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) { continue }
    foreach ($sourceFile in Get-ChildItem -LiteralPath $sourceDirectory -Recurse -File) {
        $relativeFile = $sourceFile.FullName.Substring($driveRoot.Length).TrimStart("\")
        $record = Copy-BackVerifiedFile -Source $sourceFile.FullName -RelativePath $relativeFile
        if ($null -ne $record) { $records += $record }
    }
}
foreach ($relativeReport in $relativeReports) {
    $sourceFile = Join-Path $driveRoot $relativeReport
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) { continue }
    $record = Copy-BackVerifiedFile -Source $sourceFile -RelativePath $relativeReport
    if ($null -ne $record) { $records += $record }
}

$logDir = Join-Path $workspaceRoot "resultados\logs\sincronizacion_colab_04_20x"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ("recuperacion_" + [DateTime]::Now.ToString("yyyyMMdd_HHmmss") + ".json")
[ordered]@{
    schema_version = "1.0"
    completed_at = [DateTimeOffset]::Now.ToString("o")
    source = $driveRoot
    destination = $workspaceRoot
    include_qwen = [bool]$IncludeQwen
    qwen_04_205_only = [bool]$Qwen04_205Only
    qwen_04_206_only = [bool]$Qwen04_206Only
    qwen_04_20x_only = [bool]$Qwen04_20XOnly
    deployment_only = [bool]$DeploymentOnly
    comparison_only = [bool]$ComparisonOnly
    force = [bool]$Force
    copied_files = $records
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $logPath -Encoding utf8

[pscustomobject]@{
    status = "complete"
    copied_files = $records.Count
    include_qwen = [bool]$IncludeQwen
    qwen_04_205_only = [bool]$Qwen04_205Only
    qwen_04_206_only = [bool]$Qwen04_206Only
    qwen_04_20x_only = [bool]$Qwen04_20XOnly
    deployment_only = [bool]$DeploymentOnly
    comparison_only = [bool]$ComparisonOnly
    log = $logPath
} | Format-List
