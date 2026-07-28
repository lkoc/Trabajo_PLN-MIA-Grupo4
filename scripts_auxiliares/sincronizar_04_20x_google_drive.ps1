param(
    [string]$Workspace = "D:\trabajo_PLN\Trabajo_PLN-MIA-Grupo4",
    [string]$DriveBundle = "G:\My Drive\PLN_colab_04_artifacts"
)

$ErrorActionPreference = "Stop"
$workspaceRoot = [System.IO.Path]::GetFullPath($Workspace)
$driveRoot = [System.IO.Path]::GetFullPath($DriveBundle)
$driveParent = Split-Path -Parent $driveRoot
if (-not (Test-Path -LiteralPath $workspaceRoot -PathType Container)) {
    throw "No existe el workspace: $workspaceRoot"
}
if (-not (Test-Path -LiteralPath $driveParent -PathType Container)) {
    throw "Google Drive no está disponible: $driveParent"
}
if ($driveRoot -eq $workspaceRoot) {
    throw "El bundle de Drive no puede ser el workspace."
}
New-Item -ItemType Directory -Path $driveRoot -Force | Out-Null

$requiredRelativeFiles = @(
    "datos\processed\chunks_para_etiquetar.jsonl",
    "datos\processed\taxonomia_moderacion.csv",
    "datos\processed\dataset_pseudoetiquetado_hibrido.jsonl",
    "datos\model_ready\transformer_grueso\dataset_balanceado_4a1_particionado.jsonl",
    "datos\model_ready\transformer_grueso\dataset_entrenamiento_transformer_4a1.manifest.json",
    "datos\model_ready\transformer_grueso\dataset_integrado_todas_pasadas.jsonl",
    "datos\ampliacion\ampliacion_amenaza_20260727_lote3\processed\dataset_etiquetado_utilizable.jsonl",
    "datos\ampliacion\ampliacion_dano_20260726\processed\dataset_etiquetado_utilizable.jsonl",
    "datos\ampliacion\ampliacion_dano_20260727_lote2\processed\dataset_etiquetado_utilizable.jsonl",
    "modelos\moderador_transformer_grueso\registro_modelos_comparables.json",
    "modelos\moderador_transformer_grueso\baseline_clasico_mismo_split\mejor_modelo_clasico.joblib",
    "modelos\moderador_transformer_grueso\paraphrase_minilm\best_checkpoint.pt",
    "modelos\moderador_transformer_grueso\e5_small\best_checkpoint.pt",
    "resultados\metricas\transformer_grueso\comparacion_modelos_clasicos.json",
    "resultados\metricas\transformer_grueso\evaluacion_paraphrase_minilm.json",
    "resultados\metricas\transformer_grueso\evaluacion_e5_small.json",
    "resultados\metricas\transformer_grueso\scores_paraphrase_minilm_validation.npy",
    "resultados\metricas\transformer_grueso\scores_paraphrase_minilm_test.npy",
    "resultados\metricas\transformer_grueso\scores_e5_small_validation.npy",
    "resultados\metricas\transformer_grueso\scores_e5_small_test.npy",
    "resultados\metricas\transformer_grueso\scores_clasico_ganador_validation.npy",
    "resultados\metricas\transformer_grueso\scores_clasico_ganador_test.npy"
)

function Copy-VerifiedFile {
    param([string]$RelativePath)
    $source = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $RelativePath))
    $destination = [System.IO.Path]::GetFullPath((Join-Path $driveRoot $RelativePath))
    if (-not $source.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Fuente fuera del workspace: $source"
    }
    if (-not $destination.StartsWith($driveRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Destino fuera del bundle: $destination"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Falta artefacto local: $source"
    }
    $sourceItem = Get-Item -LiteralPath $source
    $copyNeeded = -not (Test-Path -LiteralPath $destination -PathType Leaf)
    if (-not $copyNeeded) {
        $copyNeeded = (Get-Item -LiteralPath $destination).Length -ne $sourceItem.Length
    }
    if ($copyNeeded) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLower()
    $destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLower()
    if ($sourceHash -ne $destinationHash) {
        throw "Hash distinto después de copiar: $RelativePath"
    }
    $item = Get-Item -LiteralPath $destination
    return [ordered]@{
        path = $RelativePath.Replace("\", "/")
        bytes = [int64]$item.Length
        sha256 = $destinationHash
    }
}

function Copy-VerifiedSnapshotFile {
    param(
        [string]$RelativePath,
        [string]$ExpectedSha256,
        [int64]$ExpectedBytes
    )
    $source = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $RelativePath))
    $destination = [System.IO.Path]::GetFullPath((Join-Path $driveRoot $RelativePath))
    if (-not $source.StartsWith($workspaceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Fuente Qwen fuera del workspace: $source"
    }
    if (-not $destination.StartsWith($driveRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Destino Qwen fuera del bundle: $destination"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Falta archivo del checkpoint Qwen: $source"
    }
    $sourceItem = Get-Item -LiteralPath $source
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLower()
    if ($sourceItem.Length -ne $ExpectedBytes -or $sourceHash -ne $ExpectedSha256.ToLower()) {
        throw "El archivo Qwen ya no coincide con el puntero: $RelativePath"
    }
    $copyNeeded = -not (Test-Path -LiteralPath $destination -PathType Leaf)
    if (-not $copyNeeded) {
        $copyNeeded = (Get-Item -LiteralPath $destination).Length -ne $ExpectedBytes
        if (-not $copyNeeded) {
            $copyNeeded = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLower() -ne $sourceHash
        }
    }
    if ($copyNeeded) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
    $destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLower()
    if ($destinationHash -ne $sourceHash) {
        throw "Hash distinto al copiar el checkpoint Qwen: $RelativePath"
    }
    return [ordered]@{
        path = $RelativePath.Replace("\", "/")
        bytes = [int64]$ExpectedBytes
        sha256 = $destinationHash
        role = "qwen_resume_snapshot"
    }
}

function Write-PortableQwenPointer {
    param(
        [object]$Pointer,
        [string]$RelativePath,
        [string]$ExpectedSourceSha256
    )
    $source = Join-Path $workspaceRoot $RelativePath
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLower() -ne $ExpectedSourceSha256) {
        throw "El puntero Qwen cambió antes de generar su versión portable."
    }
    $Pointer.directory = ([string]$Pointer.directory).Replace("\", "/")
    foreach ($entry in $Pointer.files) {
        $entry.path = ([string]$entry.path).Replace("\", "/")
    }
    $destination = [System.IO.Path]::GetFullPath((Join-Path $driveRoot $RelativePath))
    if (-not $destination.StartsWith($driveRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Destino del puntero fuera del bundle: $destination"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    $json = $Pointer | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $destination,
        $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    return [ordered]@{
        path = $RelativePath.Replace("\", "/")
        bytes = [int64](Get-Item -LiteralPath $destination).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLower()
        role = "qwen_resume_pointer_portable"
        source_sha256 = $ExpectedSourceSha256
    }
}

function Copy-QwenResumeSnapshot {
    $pointerRelative = "modelos\qwen3_06b_lora_acoso_amenaza_4\resume_pointer.json"
    $pointerSource = Join-Path $workspaceRoot $pointerRelative
    if (-not (Test-Path -LiteralPath $pointerSource -PathType Leaf)) {
        return [ordered]@{
            available = $false
            records = @()
            epoch = $null
            completed_batches = $null
            note = "No existe un checkpoint reanudable de 04_7/04_205."
        }
    }

    # 04_7 alterna dos slots y publica el puntero al final. Solo copiamos el
    # slot activo y comprobamos que el puntero no haya cambiado durante la copia.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $pointerHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $pointerSource).Hash.ToLower()
        $pointerBytes = (Get-Item -LiteralPath $pointerSource).Length
        $pointer = Get-Content -LiteralPath $pointerSource -Raw | ConvertFrom-Json
        $snapshotRecords = @()
        try {
            foreach ($entry in $pointer.files) {
                $snapshotRecords += Copy-VerifiedSnapshotFile `
                    -RelativePath ([string]$entry.path) `
                    -ExpectedSha256 ([string]$entry.sha256) `
                    -ExpectedBytes ([int64]$entry.bytes)
            }

            $tokenizerRoot = Join-Path $workspaceRoot "modelos\qwen3_06b_lora_acoso_amenaza_4\tokenizer"
            if (-not (Test-Path -LiteralPath $tokenizerRoot -PathType Container)) {
                throw "Falta el tokenizer necesario para reanudar Qwen: $tokenizerRoot"
            }
            foreach ($tokenizerFile in Get-ChildItem -LiteralPath $tokenizerRoot -File) {
                $relative = $tokenizerFile.FullName.Substring($workspaceRoot.Length).TrimStart("\")
                $record = Copy-VerifiedFile -RelativePath $relative
                $record.role = "qwen_tokenizer"
                $snapshotRecords += $record
            }
        } catch {
            if ($attempt -eq 3) { throw }
            continue
        }

        $pointerHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $pointerSource).Hash.ToLower()
        if ($pointerHashAfter -ne $pointerHashBefore) {
            if ($attempt -eq 3) {
                throw "04_7 publicó tres checkpoints durante la copia; repita la sincronización."
            }
            continue
        }
        $snapshotRecords += Write-PortableQwenPointer `
            -Pointer $pointer `
            -RelativePath $pointerRelative `
            -ExpectedSourceSha256 $pointerHashBefore
        $pointerHashFinal = (Get-FileHash -Algorithm SHA256 -LiteralPath $pointerSource).Hash.ToLower()
        if ($pointerHashFinal -ne $pointerHashBefore) {
            if ($attempt -eq 3) {
                throw "El puntero Qwen cambió al publicarlo en Drive; repita la sincronización."
            }
            continue
        }
        return [ordered]@{
            available = $true
            records = $snapshotRecords
            epoch = [int]$pointer.epoch
            completed_batches = [int]$pointer.completed_batches
            global_optimizer_steps = [int]$pointer.global_optimizer_steps
            updated_at = [string]$pointer.updated_at
            note = "Instantánea atómica de 04_7 disponible para reanudar 04_205."
        }
    }
}

$records = @()
foreach ($relative in $requiredRelativeFiles) {
    $records += Copy-VerifiedFile -RelativePath $relative
}

$qwenResume = Copy-QwenResumeSnapshot
$records += $qwenResume.records
$qwenTraining = Join-Path $workspaceRoot "resultados\metricas\qwen3_06b_lora_acoso_amenaza_4\finetuning.json"
$qwenExportReady = Test-Path -LiteralPath $qwenTraining -PathType Leaf
$qwenFinalRecords = @()
if ($qwenExportReady) {
    $qwenFinalDirectories = @(
        "modelos\qwen3_06b_lora_acoso_amenaza_4\best_adapter",
        "resultados\metricas\qwen3_06b_lora_acoso_amenaza_4",
        "resultados\figuras\qwen3_06b_lora_acoso_amenaza_4"
    )
    foreach ($relativeDirectory in $qwenFinalDirectories) {
        $sourceDirectory = Join-Path $workspaceRoot $relativeDirectory
        if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) { continue }
        foreach ($sourceFile in Get-ChildItem -LiteralPath $sourceDirectory -Recurse -File) {
            $relative = $sourceFile.FullName.Substring($workspaceRoot.Length).TrimStart("\")
            $record = Copy-VerifiedFile -RelativePath $relative
            $record.role = "qwen_completed_artifact"
            $qwenFinalRecords += $record
        }
    }
    $qwenReport = "resultados\INFORME_QWEN_ACOSO_AMENAZA_4.md"
    if (Test-Path -LiteralPath (Join-Path $workspaceRoot $qwenReport) -PathType Leaf) {
        $record = Copy-VerifiedFile -RelativePath $qwenReport
        $record.role = "qwen_completed_artifact"
        $qwenFinalRecords += $record
    }
    $records += $qwenFinalRecords
}
$gitCommit = (git -C $workspaceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "No se pudo obtener el commit Git." }
$manifest = [ordered]@{
    schema_version = "1.0"
    created_at = [DateTimeOffset]::Now.ToString("o")
    source_workspace = $workspaceRoot
    git_commit = $gitCommit
    purpose = "inputs_shared_by_04_202_to_04_206"
    files = $records
    file_count = $records.Count
    total_bytes = [int64](($records | ForEach-Object { $_.bytes } | Measure-Object -Sum).Sum)
    qwen_local_training_complete = $qwenExportReady
    qwen_resume = $qwenResume
    qwen_completed_artifact_count = $qwenFinalRecords.Count
    qwen_note = if ($qwenExportReady) {
        "Se copiaron el best_adapter y los resultados Qwen completos; 04_205 los detectará y 04_206 podrá consumirlos."
    } else {
        $qwenResume.note
    }
}
$manifestPath = Join-Path $driveRoot "MANIFIESTO_ARTEFACTOS_04_20X.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

[pscustomobject]@{
    status = "complete"
    destination = $driveRoot
    files = $records.Count
    bytes = $manifest.total_bytes
    gib = [math]::Round($manifest.total_bytes / 1GB, 3)
    git_commit = $gitCommit
    qwen_local_training_complete = $qwenExportReady
    qwen_resume_available = $qwenResume.available
    qwen_resume_epoch = $qwenResume.epoch
    qwen_resume_completed_batches = $qwenResume.completed_batches
    manifest = $manifestPath
} | Format-List
