[CmdletBinding()]
param(
    [string] $OutputPath
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$paperRoot = Join-Path $projectRoot 'Documento_final_paper'
$sectionsRoot = Join-Path $paperRoot 'secciones'
$bibPath = Join-Path $paperRoot 'referencias.bib'

$keys = foreach ($file in Get-ChildItem -LiteralPath $sectionsRoot -Filter '*.tex' -File) {
    $content = Get-Content -Raw -Encoding UTF8 -LiteralPath $file.FullName
    $pattern = '\\(?:parencite|textcite|autocite|footcite|nocite|citep|citet|cite)(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]+)\}'
    foreach ($match in [regex]::Matches($content, $pattern)) {
        $match.Groups[1].Value -split ',' |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    }
}
$keys = $keys | Sort-Object -Unique

$bib = Get-Content -Raw -Encoding UTF8 -LiteralPath $bibPath
$entryPattern = '(?ms)^@(?<type>[A-Za-z]+)\{(?<key>[^,]+),\s*(?<body>.*?)(?=^@[A-Za-z]+\{|\z)'
$entries = @{}
foreach ($match in [regex]::Matches($bib, $entryPattern)) {
    $entries[$match.Groups['key'].Value.Trim()] = [pscustomobject]@{
        Type = $match.Groups['type'].Value.ToLowerInvariant()
        Body = $match.Groups['body'].Value
    }
}

function Get-BibField([string] $Body, [string] $Name) {
    $escapedName = [regex]::Escape($Name)
    $braced = [regex]::Match($Body, '(?m)^\s*' + $escapedName + '\s*=\s*\{(.*)\}\s*,?\s*$')
    if ($braced.Success) {
        return $braced.Groups[1].Value.Trim()
    }

    $quoted = [regex]::Match($Body, '(?m)^\s*' + $escapedName + '\s*=\s*"(.*)"\s*,?\s*$')
    if ($quoted.Success) {
        return $quoted.Groups[1].Value.Trim()
    }

    return ''
}

function Convert-LatexText([string] $Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ''
    }

    $Value = $Value -replace '[{}]', ''
    $pairs = @(
        @("\'a", ([char]0x00E1).ToString()), @("\'e", ([char]0x00E9).ToString()),
        @("\'i", ([char]0x00ED).ToString()), @("\'o", ([char]0x00F3).ToString()),
        @("\'u", ([char]0x00FA).ToString()), @("\'A", ([char]0x00C1).ToString()),
        @("\'E", ([char]0x00C9).ToString()), @("\'I", ([char]0x00CD).ToString()),
        @("\'O", ([char]0x00D3).ToString()), @("\'U", ([char]0x00DA).ToString()),
        @('\~n', ([char]0x00F1).ToString()), @('\~N', ([char]0x00D1).ToString()),
        @('\"u', ([char]0x00FC).ToString()), @('\"U', ([char]0x00DC).ToString()),
        @('\&', '&'), @('\_', '_'), @('\%', '%'), @('``', '"'), @("''", '"')
    )
    foreach ($pair in $pairs) {
        $Value = $Value.Replace($pair[0], $pair[1])
    }
    return ($Value -replace '\s+', ' ').Trim()
}

$pdfByKey = @{}
Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.pdf' -File | ForEach-Object {
    $key = $_.BaseName.Split('__')[0]
    if ($pdfByKey.ContainsKey($key)) {
        throw "Mas de un PDF local usa la clave BibTeX '$key'."
    }

    $bytes = [byte[]]::new(4)
    $stream = [System.IO.File]::OpenRead($_.FullName)
    try {
        $read = $stream.Read($bytes, 0, $bytes.Length)
    }
    finally {
        $stream.Dispose()
    }
    $signature = if ($read -eq 4) { [Text.Encoding]::ASCII.GetString($bytes) } else { '' }
    if ($signature -ne '%PDF') {
        throw "El archivo '$($_.Name)' tiene extension .pdf pero no firma %PDF."
    }

    $pdfByKey[$key] = 'referencias_y_descargas/' + $_.Name
}

$state = @{
    branez2012amixer = 'repositorio_oficial_pdf_oa_disponible_no_descargado'
    callirgos1993racismo = 'fuente_impresa_sin_pdf_oa_localizado'
    chow1970reject = 'fuente_editorial_sin_pdf_oa_localizado'
    cortes1995svm = 'pdf_editor_no_recuperado'
    cox1958logistic = 'fuente_editorial_sin_pdf_oa_localizado'
    deepseek2026v4 = 'recurso_web_oficial_sin_pdf'
    deerwester1990lsa = 'pdf_editor_descarga_bloqueada_403'
    depoix2026transcript = 'repositorio_software_web_sin_pdf'
    efron1979bootstrap = 'fuente_editorial_sin_pdf_oa_localizado'
    field2007clusterbootstrap = 'fuente_editorial_sin_pdf_oa_localizado'
    friedman2001gbm = 'pdf_editor_no_recuperado'
    grupo4dataset2026 = 'recurso_datos_versionado_sin_pdf'
    hevner2004dsr = 'repositorio_oficial_indica_documento_no_disponible'
    hf2026e5card = 'ficha_modelo_web_sin_pdf'
    hf2026minilmcard = 'ficha_modelo_web_sin_pdf'
    hf2026qwen06bcard = 'ficha_modelo_web_sin_pdf'
    lam1997majority = 'fuente_editorial_sin_pdf_oa_localizado'
    peffers2007dsrm = 'fuente_editorial_sin_pdf_oa_localizado'
    platt1999probabilistic = 'fuente_editorial_sin_pdf_oa_localizado'
    portocarrero2009racismo = 'fuente_impresa_sin_pdf_oa_localizado'
    pytorch2026bce = 'documentacion_web_oficial_sin_pdf'
    salton1988tfidf = 'pdf_repositorio_oa_descarga_bloqueada_antibot'
    sklearn2026averageprecision = 'documentacion_web_oficial_sin_pdf'
    sklearn2026groupkfold = 'documentacion_web_oficial_sin_pdf'
    sklearn2026groupshufflesplit = 'documentacion_web_oficial_sin_pdf'
    sokolova2009metrics = 'pdf_autor_descarga_bloqueada_antibot'
    thakur2025quechua = 'pdf_oa_descarga_bloqueada_403'
    unicode2025normalization = 'norma_web_oficial_sin_pdf'
    w3c2009skos = 'norma_web_oficial_sin_pdf'
    w3c2012owl2 = 'norma_web_oficial_sin_pdf'
    w3c2014turtle = 'norma_web_oficial_sin_pdf'
    wilson1927probable = 'fuente_editorial_sin_pdf_oa_localizado'
    youtube2023terms = 'terminos_servicio_web_oficial_sin_pdf'
    youtube2026sexualpolicy = 'politica_plataforma_web_oficial_sin_pdf'
    vanderlaan2007superlearner = 'pdf_autor_descarga_bloqueada_403'
    wolpert1992stacked = 'fuente_editorial_sin_pdf_oa_localizado'
    ytdlp2026 = 'repositorio_software_web_sin_pdf'
    zavala2007discurso = 'fuente_impresa_sin_pdf_oa_localizado'
    zavala2017racismo = 'repositorio_oficial_solo_epub_embargado'
}

$urlOverride = @{
    branez2012amixer = 'https://tesis.pucp.edu.pe/items/c5a89d9d-2f21-4a82-900b-4bb6121525fb'
    chow1970reject = 'https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff'
    deerwester1990lsa = 'https://www.microsoft.com/en-us/research/publication/indexing-by-latent-semantic-analysis/'
    platt1999probabilistic = 'https://www.microsoft.com/en-us/research/project/support-vector-machines/'
    prechelt1998earlystopping = 'https://publikationen.bibliothek.kit.edu/7498'
    salton1988tfidf = 'https://hdl.handle.net/1813/6721'
    zavala2017racismo = 'https://repositorio.pucp.edu.pe/items/d945348c-e161-4127-a459-8c77cfb51645'
}

$rows = foreach ($key in $keys) {
    if (-not $entries.ContainsKey($key)) {
        throw "Clave citada sin entrada BibTeX: $key"
    }

    $entry = $entries[$key]
    $title = Convert-LatexText (Get-BibField $entry.Body 'title')
    $doi = Get-BibField $entry.Body 'doi'
    $url = Get-BibField $entry.Body 'url'
    if ($urlOverride.ContainsKey($key)) {
        $url = $urlOverride[$key]
    }
    if ([string]::IsNullOrWhiteSpace($url) -and -not [string]::IsNullOrWhiteSpace($doi)) {
        $url = 'https://doi.org/' + $doi
    }

    $pdf = if ($pdfByKey.ContainsKey($key)) { $pdfByKey[$key] } else { '' }
    $status = if ($pdf) {
        'pdf_oa_validado'
    }
    elseif ($state.ContainsKey($key)) {
        $state[$key]
    }
    else {
        'sin_pdf_oa_localizado'
    }

    [pscustomobject] [ordered] @{
        clave = $key
        titulo = $title
        tipo = $entry.Type
        doi = $doi
        url = $url
        pdf_local = $pdf
        estado = $status
    }
}

if ($OutputPath) {
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $OutputPath
}
else {
    $rows | ConvertTo-Csv -NoTypeInformation
}
