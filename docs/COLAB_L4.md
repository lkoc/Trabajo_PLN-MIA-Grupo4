# Ejecución opcional en Google Colab L4

## Arquitectura

El flujo remoto no requiere Google Cloud Console, un cliente OAuth propio ni
Google Drive Desktop. Cada cuaderno Colab (`02_01` y `03_02`–`03_06b`) verifica
si Drive ya contiene el release exacto fijado al generarlo. Si falta, el mismo
bootstrap obtiene el bundle, valida su `bundle_id` y todos sus SHA-256 y lo
publica de forma inmutable. Después activa la copia verificada y reconstruye el
proyecto mínimo en `/content` antes de importar código. `02_00` permanece como
publicador manual opcional, pero ya no forma parte de la secuencia obligatoria.

Google advierte que las máquinas virtuales de Colab son efímeras y que el
montaje de Drive requiere autorización de la cuenta durante la sesión. Por eso
la fuente del bundle es reproducible, la publicación deja el manifiesto para el
final y los consumidores verifican nuevamente cada artefacto. Referencia:
[FAQ oficial de Colab](https://research.google.com/colaboratory/faq.html).

## Fuentes admitidas por todos los cuadernos Colab

La celda de bootstrap común expone dos modos:

```python
COLAB_AUTO_PUBLISH_MISSING_BUNDLE=True
COLAB_BUNDLE_SOURCE="github"       # o "local_upload"
COLAB_GITHUB_REPOSITORY="lkoc/Trabajo_PLN-MIA-Grupo4"
COLAB_GITHUB_REF="main"
```

- `github` descarga `resultados/colab_bundle` desde la rama, etiqueta o commit
  indicado. Es el modo recomendado cuando el bundle local ya fue sincronizado
  con GitHub.
- `local_upload` abre el selector del navegador. Seleccione simultáneamente los
  nueve archivos declarados: `project_core.zip`, `chunks_v2.jsonl.gz`,
  `chunks_deepseek_historicos.jsonl.gz`, los cuatro archivos históricos
  Flash/Pro, `dataset_5_salidas.jsonl.gz` y `bundle_manifest.json` desde la
  carpeta local `resultados/colab_bundle`. Este modo sirve cuando el bundle
  local es más reciente que GitHub. Colab alojado no puede leer directamente una ruta como
  `D:\trabajo_PLN\...`; el navegador es el puente.

En ambos casos se rechaza la publicación si el `bundle_id` no puede
recalcularse, el core no es compatible con el cuaderno o cualquier SHA-256 es
distinto. El generador detecta localmente si cambiaron `src/`, la configuración
o cualquiera de las entradas y reconstruye el bundle antes de emitir los
cuadernos. También puede ejecutarse explícitamente con:

```powershell
python tools/prepare_colab_bundle.py --destination resultados/colab_bundle
```

Después se sincroniza con GitHub o se selecciona `local_upload` en el propio
cuaderno consumidor.

## Publicación versionada en Drive

Cuando falta el release esperado, el bootstrap solicita la autorización
integrada de Drive y escribe:

```text
MyDrive/ModeracionPeru_Colab/
├── bundle/                         # copia activa administrada por consumidores
└── bundle_releases/
    ├── latest.json                 # puntero actualizado al final
    └── <bundle_id>/
        ├── project_core.zip
        ├── chunks_v2.jsonl.gz
        ├── chunks_deepseek_historicos.jsonl.gz
        ├── deepseek_flash_historico.jsonl.gz
        ├── deepseek_pro_historico_principal.jsonl.gz
        ├── deepseek_pro_historico_umbral.jsonl.gz
        ├── deepseek_pro_historico_sospechosos.jsonl.gz
        ├── dataset_5_salidas.jsonl.gz
        └── bundle_manifest.json    # copiado después de los artefactos
```

Si la versión ya existe, se verifica y reutiliza sin modificarla. Si no existe,
se copia primero a un directorio parcial, se valida y se promueve. Solo después
se reemplaza `latest.json`, que registra `bundle_id`, SHA-256 del core y SHA-256
del manifiesto. Una carpeta existente pero inválida bloquea el proceso en vez de
ser sobrescrita.

La publicación solo se ejecuta si el release esperado falta o es inválido. Si
ya existe y supera la verificación, el cuaderno lo reutiliza sin volver a
descargar ni copiar sus archivos.

## Consumidores y GPU

Los cuadernos `02_01` y `03_02`–`03_06b` pueden ejecutarse en Colab. `02_01` usa la API DeepSeek y funciona con un runtime CPU; no reserva una GPU innecesaria. `03_02`–`03_04` requieren L4; `03_05`, `03_06` y el toy `03_06b` recomiendan A100. Su bootstrap:

1. monta Drive y busca el release exacto fijado por el cuaderno;
2. si falta, lo descarga de GitHub —o abre `local_upload`— y lo publica atómicamente;
3. exige que el SHA-256 del core coincida con el esperado por el cuaderno;
4. valida el manifiesto y todos los archivos de la versión;
5. actualiza la copia activa `bundle/` dejando su manifiesto para el final;
6. extrae e instala el core en el SSD efímero `/content`;
7. exige una `NVIDIA L4` solo si el cuaderno declara `requires_cuda=true` y `COLAB_REQUIRE_L4=True`; los cuadernos Qwen usan `False`, exigen CUDA y activan automáticamente el perfil BF16 de 40 GB en A100/H100/H200 o hardware equivalente.

Los cuadernos `02_02`, `03_05`, `03_06` y `03_06b` están optimizados para una A100 de 40 GB. La detección se basa en CUDA, soporte BF16 y memoria observable (al menos 39 GB), no únicamente en el nombre comercial. En `03_05` y `03_06` el lote efectivo permanece en ocho (`8×1` en A100 frente a `2×4` en L4). El toy `03_06b` usa secuencias de 1.536 tokens, LoRA causal, lote 8 en A100 y generación restringida por lotes de 16. Solo entrena con 800 filas y evalúa 200 de validation más 200 de test.

Los entrenamientos `03_02`–`03_06` se ejecutan sobre el SSD efímero. Los perfiles
reanudables reflejan automáticamente cada checkpoint que termina de escribir `Trainer` en
`COLAB_CONTEXT.drive_run_dir/trainer_checkpoints/<firma-del-run>/<trainer>/`.
Cada versión se conserva con un nombre inmutable
`checkpoint-<step>-<sha16>.tar`, un manifiesto independiente y SHA-256;
`latest.json` solo cambia después de verificar la copia completa en Drive. El TAR
contiene pesos o adaptadores, optimizador, scheduler, RNG y
`trainer_state.json`, por lo que un checkpoint guardado al terminar la época 2
reanuda la época 3 con `resume_from_checkpoint`, sin reiniciar el ajuste. Las
cascadas mantienen separados sus trainers. `03_06b` también entrena en el SSD,
pero no usa la infraestructura de candidatos/checkpoints de esas familias: una
firma que combina dataset, Markdown, semilla e hiperparámetros evita repetir una
corrida completa ya materializada.

Al activar esta función también se migra inmediatamente el checkpoint local más
nuevo que todavía no exista en Drive. Tras perder o reiniciar el kernel, la
siguiente ejecución verifica y restaura al SSD el checkpoint persistente más
nuevo. Si el último archivo estuviera incompleto, intenta el anterior verificado.
La publicación de resultados es automática al completar el entrenamiento;
`03_02` publica MiniLM antes de comenzar E5, vuelve a publicar al terminar E5 y
publica de nuevo si se ejecuta la evaluación por canal. La campaña MiniLM
mejorada publica también cada variante 192/256, semilla o ablación focal antes
de iniciar la siguiente. `03_06b` permite copiar manualmente a Drive su dataset,
adaptador, predicciones, métricas y reporte. Esa copia conserva el carácter
independiente y nunca se restaura como entrada de `03_07`.

`03_06` implementa además warm-start entre cuadernos. Antes del barrido
estructurado, restaura desde
`runs/03_05/03_05_working_v2_1` la ranura publicada más reciente cuyo TAR y
SHA-256 sean válidos, en un directorio auxiliar del SSD. El selector vuelve a
comprobar el `candidate.json`, el manifiesto del checkpoint, el SHA-256 del
dataset, las 22 salidas y la longitud de contexto antes de cargar el adaptador.
La copia auxiliar nunca reemplaza el run activo de `03_06` ni se incluye en su
publicación final. Si `03_05` se ejecutó con otro `COLAB_RUN_ID`, actualice
`QWEN_LORA_PARENT_RUN_ID` explícitamente.

Si el bundle cambió después de que el kernel importó `moderacion_peru`, el
cuaderno exige reiniciar el kernel para evitar mezclar versiones. Los modelos se
descargan al caché efímero `/content/huggingface`; no se sincronizan videos,
audio, VTT, PDFs, modelos Ollama ni cachés de Hugging Face.

Los modelos Hugging Face configurados son públicos. El bootstrap define
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` y `HF_HOME=/content/huggingface` antes de
importar la biblioteca, evitando que un kernel controlado desde VS Code intente
consultar el almacén de secretos exclusivo de la interfaz web de Colab. Un
repositorio privado o restringido requeriría autenticación explícita y segura.
Consulte las [variables de entorno oficiales de Hugging Face](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

La credencial comercial de `02_01` se configura como secreto de Colab llamado
`DEEPSEEK_API_KEY`. El cuaderno intenta leerlo únicamente cuando la variable de
entorno no existe. `RUN_API_PREFLIGHT=True` consulta `/models` y saldo sin enviar
textos; `RUN_CALIBRATION`, `RUN_PRIMARY` y `RUN_DIRECTED_REVIEW` sí transmiten
los chunks al proveedor. La barra registra velocidad, caché, costo por tokens y
saldo periódico; los topes se configuran antes de activar cada fase.

Los checkpoints de entrenamiento y los resultados finales ocupan espacios
distintos en Drive:

```text
MyDrive/ModeracionPeru_Colab/runs/<notebook_id>/<run_id>/
├── trainer_checkpoints/<firma-del-run>/<trainer>/
│   ├── checkpoint-<step>-<sha16>.tar
│   ├── checkpoint-<step>-<sha16>.json
│   └── latest.json
├── publications/
│   ├── run_outputs-a.tar
│   ├── run_outputs-b.tar
│   ├── run_manifest-a.json
│   └── run_manifest-b.json
└── run_manifest.json                 # puntero activo
```

La publicación final usa TAR sin compresión: los pesos ya comprimidos apenas se
reducían con gzip y `03_02` podía quedar interrumpido durante horas al recomprimir
varios GB. No duplica los directorios `trainer`, porque sus checkpoints completos
ya están en `trainer_checkpoints`. Escribe siempre en la ranura inactiva, relee y
verifica tamaño y SHA-256 desde Drive y solo entonces cambia
`run_manifest.json`. Si la copia activa estuviera truncada, la recuperación prueba
la ranura anterior. Las publicaciones históricas `run_outputs.tar.gz` siguen
siendo compatibles y se conservan como respaldo durante la migración.

En `02_01`, `AUTO_PUBLISH_CHECKPOINTS=True` publica automáticamente después de
la recuperación histórica, cada diez ventanas de procesamiento, al cerrar una
fase y ante `Ctrl+C`. Cada grupo de cinco respuestas se fuerza antes a disco con
`fsync`, por lo que la siguiente ejecución restaura el TAR.GZ y omite todos los
`chunk_id` ya válidos. En los cuadernos `03_x`, la celda
`PUBLISH_TO_DRIVE=True` queda como reintento manual opcional; no es necesaria tras
una terminación normal.

`03_01`, `03_08` y `03_07a` son adecuados para CPU local. `03_07` también usa
CPU, pero debe abrirse en Colab web porque restaura allí los candidatos publicados
en Drive. Activar Colab no inventa una corrida: los interruptores permanecen en
`False` hasta completar el preflight correspondiente.

El reporte `03_07a` no monta Drive ni usa Google Drive para escritorio. Consulta
la Google Drive API desde el kernel local mediante OAuth de solo lectura. La
primera ejecución requiere un cliente OAuth de escritorio en
`config/google_drive_oauth_client.json` y consentimiento en el navegador; el
token renovable se guarda en `.secrets/`, fuera de Git. Después compara fecha y
SHA-256 con `sincronizacion_google_drive_03_07.json`, descarga solo una publicación
nueva o diferente y extrae exclusivamente los JSON de resultados. No descarga
modelos ni checkpoints de `Trainer`.

La apertura de test de `03_07` acepta candidatos PEFT con cinco salidas primarias
y cualquier número de auxiliares opcionales. Para restaurar pesos crea la cabeza
con la dimensión declarada o inferida del adaptador; para las métricas y el
ensemble conserva solo las primeras cinco en el orden de la taxonomía. Después
de cada miembro persiste scores parciales firmados para reanudar exactamente la
misma apertura si falla otro miembro. Esos scores no pueden intervenir en
selección, calibración ni cambio de política.

## Secuencia práctica

1. Genere el cuaderno requerido; el generador reconstruirá
   `resultados/colab_bundle` automáticamente si detecta cambios.
2. Sincronice el bundle y los cuadernos con GitHub si conservará
   `COLAB_BUNDLE_SOURCE="github"`.
3. Abra `02_01` o `03_02`–`03_06b`, seleccione Colab/L4 y ejecute desde la primera
   celda.
4. Autorice `drive.mount()`. Si el release falta, el mismo cuaderno lo publicará.
5. Confirme **Bundle de Colab verificado** antes de activar el procesamiento.

No hay credenciales, tokens ni IDs de carpetas personales versionados en el
proyecto.
