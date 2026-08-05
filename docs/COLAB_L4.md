# Ejecución opcional en Google Colab L4 desde VS Code

## Decisión de arquitectura

El flujo usa la extensión oficial **Google Colab** para VS Code y Google Drive como único transporte. No necesita GitHub ni clona el repositorio. Los `.ipynb` permanecen en el workspace local; la celda integrada monta Drive, verifica el bundle, reconstruye un proyecto mínimo en `/content` y exige una GPU `NVIDIA L4`.

La extensión oficial se instala como `google.colab`; su flujo es `Select Kernel > Colab > Auto Connect`. `drive.mount()` funciona desde la extensión a partir de v0.2.1. Google advierte que montar Drive concede al código acceso a sus archivos y que la VM y las librerías instaladas no forman parte del notebook compartido, razón por la que cada cuaderno contiene un bootstrap explícito y verificable.

Referencias: [extensión oficial en VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=google.colab), [anuncio de Google](https://developers.googleblog.com/en/google-colab-is-coming-to-vs-code/), [limitaciones conocidas de la extensión](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds) y [FAQ de Colab](https://research.google.com/colaboratory/faq.html).

## Cuellos de botella identificados

| Componente | Cuello de botella | Decisión |
|---|---|---|
| `02_01` Ollama | En esta máquina Ollama usó 100 % CPU; el piloto completo se proyectó en 14,4 h | Mantener Ollama local y añadir Hugging Face/Qwen 4B sobre L4 como alternativa, con el mismo esquema y reanudación |
| carga de Drive | Acceder repetidamente a JSONL grandes mediante FUSE es lento y está sujeto a cuotas | Comprimir una vez, verificar SHA-256 y copiar cada entrada al SSD `/content` |
| descarga de modelos | Los backbones ocupan GB y cambian con cada familia | Descargar desde Hugging Face al caché efímero `/content/huggingface`; no sincronizar el caché |
| `03_01` clásicos | TF-IDF y scikit-learn consumen CPU/RAM; CUDA no los acelera | Ejecutar localmente; no transferir datos a Colab solo por disponer de L4 |
| `03_02` encoders | tokenización, entrenamiento e inferencia de 117 244 filas | Colab L4 recomendado |
| `03_03` cascada | dos etapas y regeneración repetida de scores | Colab L4 recomendado; publicar scores/checkpoints consistentes |
| `03_04` multitarea | mayor memoria de activaciones y varias cabezas | L4, AMP/BF16 si el modelo lo admite y acumulación de gradiente |
| `03_05` Qwen-LoRA | principal presión de VRAM y checkpoints | L4 recomendado; guardar en `/content` y publicar por checkpoint |
| `03_06` Qwen estructurado | entrenamiento/inferencia generativa | L4 recomendado |
| `03_07`–`03_08` | métricas y auditoría sobre predicciones ya materializadas | CPU local; usar L4 solo si es necesario regenerar inferencias |
| escritura de checkpoints | muchos archivos pequeños directamente en Drive son lentos y vulnerables a interrupciones | empaquetar el run en un TAR.GZ local, copiarlo como `.partial`, verificar y renombrar antes del manifiesto |

Los cuadernos continúan con `RUN=False`: habilitar Colab no inventa una corrida ni métricas. Primero se ejecuta un smoke test y después se activa el entrenamiento correspondiente.

## Contenido sincronizado

La carpeta privada ya creada es [ModeracionPeru_Colab](https://drive.google.com/drive/folders/1o4qLzd6BoPRhaI4W22Vyw0wcMmPD3B_5). Su subcarpeta `bundle` contiene exactamente:

| Archivo | Tamaño | Consumidores |
|---|---:|---|
| `project_core.zip` | 41 372 B | todos los cuadernos Colab |
| `chunks_v2.jsonl.gz` | 11 549 973 B | `02_01` |
| `dataset_5_salidas.jsonl.gz` | 21 275 598 B | `03_02`–`03_06` |
| `bundle_manifest.json` | 2 781 B | verificación de todo el bundle |

Total: aproximadamente 32,9 MB en lugar de 147,2 MB sin comprimir. No se sincronizan videos, audio, transcripciones crudas, PDFs, archivo histórico, modelos Ollama, caché de Hugging Face, paper, presentación ni frontends. La carpeta no está compartida públicamente.

## Preparación o actualización local

El bundle reproducible se conserva también en `resultados/colab_bundle`:

```powershell
python tools/prepare_colab_bundle.py --destination resultados/colab_bundle
```

Si se instala Google Drive para escritorio, puede actualizarse directamente mediante:

```powershell
.\tools\sync_colab_drive.ps1 -DriveRoot "RUTA\My Drive\ModeracionPeru_Colab"
```

El manifiesto se escribe al final. En la interfaz web o mediante el conector de Drive, se deben **reemplazar** los cuatro archivos existentes, no crear duplicados con el mismo nombre.

## Ejecución desde VS Code

1. Instale las extensiones `Jupyter` y `Google Colab` de Google.
2. Abra uno de los cuadernos habilitados: `02_01` o `03_02`–`03_06`.
3. Seleccione `Select Kernel > Colab > Auto Connect`, autentíquese y elija GPU L4.
4. Ejecute la celda “Backend opcional Google Colab L4”. Si el montaje no se abre, use `Colab: Mount Google Drive to Server...` en la paleta de comandos.
5. Confirme `backend=cuda` y `device_name=NVIDIA L4`. El cuaderno se detiene explícitamente ante CPU, T4 u otra GPU mientras `COLAB_REQUIRE_L4=True`.
6. Mantenga `RUN=False` para el preflight; después active un límite pequeño. Use un `COLAB_RUN_ID` nuevo para otro experimento o deje vacío para reanudar `<cuaderno>_working_v2_1`.
7. Tras un checkpoint consistente, cambie `PUBLISH_TO_DRIVE=True`. La siguiente sesión restaurará el TAR.GZ verificado antes de continuar.

Los secretos no se guardan en Drive ni en los cuadernos. Los modelos públicos actuales no requieren token; si en el futuro se usa un repositorio restringido, el token debe configurarse en los secretos de Colab.

## Salidas y recuperación

Cada corrida se publica bajo:

```text
MyDrive/ModeracionPeru_Colab/runs/<notebook_id>/<run_id>/
├── run_outputs.tar.gz
└── run_manifest.json
```

El manifiesto registra contrato, hardware, fecha, tamaño y SHA-256. No se registra un modelo para producción hasta recuperar sus artefactos, validar las cinco salidas y completar las métricas con el protocolo de validation/test.
