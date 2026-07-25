# Instructivo de LM Studio para este proyecto

## Estado reparado el 19 de julio de 2026

Se encontró un CLI antiguo funcional, pero configurado para iniciar una aplicación eliminada en `C:\LM Studio\LM Studio.exe`. Se realizaron estas acciones:

- instalación oficial de LM Studio Desktop `0.4.19+2` mediante `winget`;
- corrección del puntero interno hacia `C:\Users\USER\AppData\Local\Programs\LM Studio\LM Studio.exe`;
- instalación del daemon oficial `llmster 0.0.19+2`;
- conservación de `.lmstudio`, configuraciones y modelos anteriores;
- inicio y verificación del daemon;
- verificación de la API OpenAI-compatible en `http://localhost:1234/v1/models`.
- descarga de `qwen/qwen3.5-9b` GGUF `Q4_K_M` (6.55 GB);
- selección del runtime CPU AVX2 `llama.cpp-win-x86_64-avx2@2.24.0`, porque el runtime Vulkan no detectó UUIDs válidos en estas Radeon;
- desactivación predeterminada de `enableThinking` para este artefacto local;
- carga con el identificador estable `qwen-local-primary` y contexto 16K;
- prueba real de salida estructurada: HTTP 200, JSON válido, 100 s en frío y 26 s con el prefijo reutilizado.

La interfaz de escritorio recién instalada se cerró durante la prueba automática inicial, pero el daemon programático quedó operativo. Para este cuaderno se recomienda usar `llmster`, porque no depende de mantener abierta la GUI.

## Verificación rápida

Abre PowerShell y ejecuta:

```powershell
lms daemon up
lms daemon status
lms status
Invoke-RestMethod http://localhost:1234/v1/models
```

El estado esperado es:

```text
llmster ... is running
Server: ON (port: 1234)
```

Si el servidor estuviera apagado:

```powershell
lms server start --port 1234 --bind 127.0.0.1
```

No uses `--bind 0.0.0.0` ni `--cors` para este proyecto: expondrían el servidor a otros equipos o páginas web. El cuaderno solo necesita `localhost`.

## Descargar los modelos candidatos

Primero inicia el daemon. Después usa selección interactiva para confirmar una cuantización GGUF Q4 compatible:

```powershell
lms get qwen/qwen3.5-9b --gguf --select
lms get google/gemma-4-12b --gguf --select
```

Si el nombre virtual de Gemma 4 todavía no se resuelve en LM Studio, búscalo en la pestaña **Discover** con `Gemma 4 12B Instruct GGUF`, o ejecuta:

```powershell
lms get "Gemma 4 12B Instruct" --gguf --select
```

Selecciona preferentemente `Q4_K_M` o la variante Q4 recomendada por LM Studio. Evita descargar varias cuantizaciones del mismo modelo hasta medir la primera.

Alternativas:

```powershell
lms get "Qwen3 8B Instruct" --gguf --select
lms get "Qwen3.5 4B Instruct" --gguf --select
```

El modelo 4B es útil para probar el pipeline, pero no se recomienda como anotador final único.

## Inspeccionar y estimar memoria

```powershell
lms ls
lms load --estimate-only <model-key> --context-length 16384 --gpu off
```

En este equipo empieza con contexto `16384`. El prompt normativo completo más cuatro chunks debería caber, y un contexto menor reduce RAM y KV cache. Si el preflight informa que la solicitud no cabe, baja `BATCH_SIZE` a 2 o 1 antes de aumentar el contexto.

## Cargar con identificador estable

Ejemplo para el modelo primario:

```powershell
lms runtime select llama.cpp-win-x86_64-avx2@2.24.0
lms load <model-key-qwen> --context-length 16384 --gpu off --parallel 1 --identifier qwen-local-primary -y
```

Ejemplo para el revisor:

```powershell
lms unload --all
lms load <model-key-gemma> --context-length 16384 --gpu off --parallel 1 --identifier gemma-local-review -y
```

Luego comprueba:

```powershell
lms ps
Invoke-RestMethod http://localhost:1234/v1/models
```

Usa exactamente `qwen-local-primary` o `gemma-local-review` en la configuración del cuaderno.

### CPU frente a GPU AMD

En esta máquina el runtime Vulkan `2.24.0` falló con `Vulkan survey did not provide device UUIDs`. El setup reparado usa explícitamente CPU AVX2:

```powershell
lms unload --all
lms runtime select llama.cpp-win-x86_64-avx2@2.24.0
lms load <model-key> --context-length 16384 --gpu off --parallel 1 --identifier qwen-local-primary -y
```

La RX 570 tiene solo 4 GB de VRAM y el modelo requiere cerca de 6.10 GiB. No vuelvas a seleccionar Vulkan ni fuerces `--gpu max` hasta que una versión posterior del runtime reconozca correctamente las GPU. Para comprobar una actualización, ejecuta `lms runtime update` y luego `lms runtime survey`.

### Desactivar razonamiento en Qwen3.5

El artefacto virtual de Qwen activa razonamiento por defecto. Para clasificación masiva quedó cambiado a `false` en:

```text
C:\Users\USER\.lmstudio\hub\models\qwen\qwen3.5-9b\model.yaml
```

La sección esperada es:

```yaml
customFields:
  - key: enableThinking
    defaultValue: false
```

Si una actualización del artefacto lo restablece a `true`, vuelve a cambiarlo y recarga el modelo. Con razonamiento activo, el modelo puede consumir todo `max_tokens` antes de producir el JSON.

## Probar la API

```powershell
$body = @{
  model = 'qwen-local-primary'
  messages = @(@{role='user'; content='Responde únicamente: OK'})
  temperature = 0
  max_tokens = 8
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri 'http://localhost:1234/v1/chat/completions' `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body
```

## Operación cotidiana

Inicio:

```powershell
lms daemon up
lms server start --port 1234 --bind 127.0.0.1
lms runtime select llama.cpp-win-x86_64-avx2@2.24.0
lms load qwen/qwen3.5-9b --context-length 16384 --gpu off --parallel 1 --identifier qwen-local-primary -y
```

Diagnóstico:

```powershell
lms daemon status
lms status
lms ps
lms log stream --source server
```

Liberar memoria al terminar:

```powershell
lms unload --all
```

Apagar completamente el servicio:

```powershell
lms server stop
lms daemon down
```

## Reparación si vuelve a aparecer `ENOENT`

Comprueba primero la instalación:

```powershell
winget list --id ElementLabs.LMStudio --exact
Get-Command lms -All
```

La ruta actual de la aplicación es:

```text
C:\Users\USER\AppData\Local\Programs\LM Studio\LM Studio.exe
```

Para reinstalar sin borrar modelos:

```powershell
winget install --id ElementLabs.LMStudio --exact --source winget
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://lmstudio.ai/install.ps1 | iex"
lms daemon up
```

Los modelos no se guardan junto al ejecutable. El daemon reparado descargó Qwen en `C:\Users\USER\.lmstudio\models`; la configuración antigua de la GUI conserva `D:\lmstudio_models`, pero llmster no la usó para esta descarga.

## Problemas frecuentes

- **`Connection refused`**: ejecuta `lms daemon up` y `lms server start`.
- **`No Models Loaded`**: descarga y carga el modelo; luego copia su identificador al cuaderno.
- **Falta de memoria**: reduce `BATCH_SIZE`, usa contexto 16384/12288, prueba Q4 o cambia a un modelo 8–9B.
- **JSON semánticamente inválido**: conserva temperatura 0, no elimines el JSON Schema y revisa el reporte de reintentos.
- **Muy lento**: mide primero 20–50 chunks y baja `BATCH_SIZE` si el contexto crece. La aceleración Vulkan actual no funciona en esta máquina; usa CPU hasta actualizar y volver a ejecutar el sondeo.
- **La GUI no abre**: el cuaderno puede trabajar con `llmster`; no es necesario detener el etiquetado. Para una reparación visual posterior, cierra el daemon, abre el acceso directo de LM Studio y revisa los logs en `%APPDATA%\LM Studio\logs`.
