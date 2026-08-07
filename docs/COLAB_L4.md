# Ejecución opcional en Google Colab L4 desde VS Code

## Decisión de arquitectura

El flujo de ejecución remota usa la extensión oficial **Google Colab** para VS Code y **solo Google Drive** como transporte hacia la VM. No clona ni descarga el repositorio desde GitHub. `02_00_preparacion_bundle_colab.ipynb`, ejecutado con kernel local, prepara una versión inmutable bajo `bundle_releases/<bundle_id>`; los `.ipynb` remotos permanecen en el workspace local y su celda integrada monta Drive, activa exactamente esa versión, reconstruye un proyecto mínimo en `/content` y exige una GPU `NVIDIA L4`.

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
| `03_06` Qwen estructurado | entrenamiento discriminativo con penalización `SEGURO+daño` | L4 recomendado |
| `03_07`–`03_08` | métricas y auditoría sobre predicciones ya materializadas | CPU local; usar L4 solo si es necesario regenerar inferencias |
| escritura de checkpoints | muchos archivos pequeños directamente en Drive son lentos y vulnerables a interrupciones | empaquetar el run en un TAR.GZ local, copiarlo como `.partial`, verificar y renombrar antes del manifiesto |

Los cuadernos continúan con `RUN_TRAINING=False`: habilitar Colab no inventa una corrida ni métricas. Primero se ejecuta un smoke test y después se activa el entrenamiento correspondiente. Al activarlo, cada rama completa fit, calibración, test, checkpoint y candidato; una firma ya terminada devuelve no-op.

## Contenido sincronizado

La carpeta privada ya creada es [ModeracionPeru_Colab](https://drive.google.com/drive/folders/1o4qLzd6BoPRhaI4W22Vyw0wcMmPD3B_5). Cada versión completa contiene exactamente:

| Archivo | Tamaño | Consumidores |
|---|---:|---|
| `project_core.zip` | 125 923 B | todos los cuadernos Colab |
| `chunks_v2.jsonl.gz` | 43 159 648 B | `02_01` |
| `dataset_5_salidas.jsonl.gz` | 21 201 195 B | `03_01`–`03_08` |
| `bundle_manifest.json` | 3 070 B | identidad y verificación de todo el bundle |

Total: aproximadamente 61,5 MiB. No se sincronizan videos, audio, transcripciones crudas, PDFs, archivo histórico, modelos Ollama, caché de Hugging Face, paper, presentación ni frontends. La carpeta de Drive no está compartida públicamente.

La estructura de transporte es:

```text
ModeracionPeru_Colab/
├── bundle/                         # copia activa; Colab puede reemplazarla
└── bundle_releases/
    ├── latest.json                 # puntero publicado al final por 02_00
    └── <bundle_id>/                # versión inmutable publicada por 02_00
        ├── project_core.zip
        ├── chunks_v2.jsonl.gz
        ├── dataset_5_salidas.jsonl.gz
        └── bundle_manifest.json
```

El `bundle_id` se calcula con los SHA-256 del core y de las entradas, no con la fecha de creación. Por ello, repetir `02_00` sin cambiar contenido devuelve `already_present`. `02_00` actualiza `latest.json` después de verificar la versión completa. Cada cuaderno Colab lee ese puntero, comprueba que el core sea compatible con el cuaderno y, si `bundle/` está atrasado, valida todos los archivos de `bundle_releases/<bundle_id>`, los copia y reemplaza el manifiesto activo al final. Así, un snapshot nuevo publicado después de `02_05` no obliga a editar manualmente los cuadernos mientras el código no haya cambiado. Si el kernel ya importó código anterior, exige reiniciarlo antes de continuar.

## Preparación o actualización local

La vía recomendada es abrir `flujo/02_etiquetado/02_00_preparacion_bundle_colab.ipynb` con **kernel local**, configurar:

```python
DRIVE_ROOT=Path("RUTA_LOCAL_GOOGLE_DRIVE")/"ModeracionPeru_Colab"
RUN_PREPARE_BUNDLE=True
```

y ejecutar sus celdas. El cuaderno muestra una barra de avance, regenera `resultados/colab_bundle`, verifica cada SHA-256 y copia la versión a `bundle_releases/<bundle_id>`. Debe esperarse a que Google Drive Desktop termine la sincronización antes de cambiar al kernel Colab.

La operación equivalente por terminal es:

```powershell
.\tools\sync_colab_drive.ps1 -DriveRoot "RUTA\My Drive\ModeracionPeru_Colab"
```

La compresión gzip usa nivel 9, nombre interno vacío y `mtime=0`, de modo que el
mismo contenido produce un archivo reproducible. En una máquina recién clonada:

```powershell
python tools/restore_synced_checkpoints.py
```

restaura las entradas a sus rutas locales después de comprobar el SHA-256 del
archivo comprimido y del contenido. Además, cada cuaderno `03_01`–`03_08`
verifica el dataset antes de usarlo: lo descomprime solo si falta y falla si una
copia existente no coincide. El dataset actual no necesita particionarse; si su
gzip se acerca a 45–50 MiB, el contrato deberá ampliarse a archivos por `split`
y partes numeradas.

Ejecute `02_00` en dos momentos: después de `01_03`, antes de `02_01`, para publicar los chunks; y después de que `02_05` publique un snapshot nuevo, antes de iniciar `03`, para publicar el dataset. La divergencia bloquea el consumo de un dataset distinto del declarado. No se reemplazan manualmente archivos dentro de una versión ya publicada: cualquier cambio produce otro `bundle_id`.

## Ejecución desde VS Code

1. Instale las extensiones `Jupyter` y `Google Colab` de Google.
2. Ejecute primero `02_00` con kernel local y espere la confirmación de sincronización de Drive.
3. Abra uno de los cuadernos habilitados: `02_01` o `03_02`–`03_06`.
4. Seleccione `Select Kernel > Colab > Auto Connect`, autentíquese y elija GPU L4.
5. Ejecute la celda “Backend opcional Google Colab L4”. Si el montaje no se abre, use `Colab: Mount Google Drive to Server...` en la paleta de comandos.
6. Confirme que aparece **Bundle de Colab verificado**, junto con el `bundle_id`, `backend=cuda` y `device_name=NVIDIA L4`. El cuaderno se detiene explícitamente ante una versión ausente, SHA inválido, CPU, T4 u otra GPU mientras `COLAB_REQUIRE_L4=True`.
7. Mantenga `RUN_TRAINING=False` para el preflight; después actívelo. Use un `COLAB_RUN_ID` nuevo para otro experimento o deje vacío para reanudar `<cuaderno>_working_v2_1`.
8. Tras un checkpoint consistente, cambie `PUBLISH_TO_DRIVE=True`. La siguiente sesión restaurará el TAR.GZ verificado antes de continuar.

Los secretos no se guardan en Drive ni en los cuadernos. Los modelos públicos actuales no requieren token; si en el futuro se usa un repositorio restringido, el token debe configurarse en los secretos de Colab.

## Salidas y recuperación

Cada corrida se publica bajo:

```text
MyDrive/ModeracionPeru_Colab/runs/<notebook_id>/<run_id>/
├── run_outputs.tar.gz
└── run_manifest.json
```

El manifiesto registra contrato, hardware, fecha, tamaño y SHA-256. Los candidatos usan rutas relativas al directorio del run, de modo que el TAR.GZ puede extraerse bajo `modelos/v2/<familia>/` sin conservar rutas `/content/...`.

Para comparar localmente:

1. descargue o restaure `run_outputs.tar.gz`;
2. verifique el SHA-256 de `run_manifest.json`;
3. extraiga su contenido bajo una carpeta propia de `modelos/v2`;
4. compruebe que `candidate.json` y `checkpoint_manifest.json` quedan juntos;
5. ejecute `03_07` o `modperu publish-model`.

No se registra un modelo para producción hasta recuperar sus artefactos, validar las cinco salidas y completar el protocolo de validation/test. `03_07` rechaza automáticamente candidatos de otro SHA-256 de dataset.
