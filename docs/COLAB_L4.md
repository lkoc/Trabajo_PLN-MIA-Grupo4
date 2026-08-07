# Ejecución opcional en Google Colab L4

## Arquitectura

El flujo remoto no requiere Google Cloud Console, un cliente OAuth propio ni
Google Drive Desktop. `02_00_preparacion_bundle_colab.ipynb` se ejecuta
directamente en Google Colab, obtiene un bundle ya construido, verifica su
`bundle_id` y todos sus SHA-256, monta Drive mediante `drive.mount()` y publica
una versión inmutable. Después, `02_01` y `03_02`–`03_06` resuelven
`bundle_releases/latest.json`, validan esa versión y reconstruyen el proyecto
mínimo en `/content` antes de importar código.

Google advierte que las máquinas virtuales de Colab son efímeras y que el
montaje de Drive requiere autorización de la cuenta durante la sesión. Por eso
la fuente del bundle es reproducible, la publicación deja el manifiesto para el
final y los consumidores verifican nuevamente cada artefacto. Referencia:
[FAQ oficial de Colab](https://research.google.com/colaboratory/faq.html).

## Fuentes admitidas por `02_00`

La celda de configuración expone dos modos:

```python
RUN_PUBLISH_BUNDLE=False
BUNDLE_SOURCE="github"       # o "local_upload"
GITHUB_REPOSITORY="lkoc/Trabajo_PLN-MIA-Grupo4"
GITHUB_REF="main"
```

- `github` descarga `resultados/colab_bundle` desde la rama, etiqueta o commit
  indicado. Es el modo recomendado cuando el bundle local ya fue sincronizado
  con GitHub.
- `local_upload` abre el selector del navegador. Seleccione simultáneamente
  `project_core.zip`, `chunks_v2.jsonl.gz`,
  `dataset_5_salidas.jsonl.gz` y `bundle_manifest.json` desde la carpeta local
  `resultados/colab_bundle`. Este modo sirve cuando el bundle local es más
  reciente que GitHub. Colab alojado no puede leer directamente una ruta como
  `D:\trabajo_PLN\...`; el navegador es el puente.

En ambos casos se rechaza la publicación si el `bundle_id` no puede
recalcularse, el core no es compatible con el cuaderno o cualquier SHA-256 es
distinto. `02_00` no construye datos dentro de la VM: el bundle local se genera,
cuando corresponda, con:

```powershell
python tools/prepare_colab_bundle.py --destination resultados/colab_bundle
```

Después se sincroniza con GitHub o se usa `local_upload`.

## Publicación versionada en Drive

Al activar `RUN_PUBLISH_BUNDLE=True`, Colab solicita la autorización integrada
de Drive y escribe:

```text
MyDrive/ModeracionPeru_Colab/
├── bundle/                         # copia activa administrada por consumidores
└── bundle_releases/
    ├── latest.json                 # puntero actualizado al final
    └── <bundle_id>/
        ├── project_core.zip
        ├── chunks_v2.jsonl.gz
        ├── dataset_5_salidas.jsonl.gz
        └── bundle_manifest.json    # copiado después de los artefactos
```

Si la versión ya existe, se verifica y reutiliza sin modificarla. Si no existe,
se copia primero a un directorio parcial, se valida y se promueve. Solo después
se reemplaza `latest.json`, que registra `bundle_id`, SHA-256 del core y SHA-256
del manifiesto. Una carpeta existente pero inválida bloquea el proceso en vez de
ser sobrescrita.

Ejecute `02_00` dos veces en el recorrido completo: después de `01_03`, antes de
`02_01`, para publicar los chunks; y después de que `02_05` cree un snapshot
nuevo, antes de entrenar con `03_02`–`03_06`.

## Consumidores y GPU

Los cuadernos `02_01` y `03_02`–`03_06` pueden ejecutarse en Colab. Su bootstrap:

1. monta Drive y lee `bundle_releases/latest.json`;
2. exige que el SHA-256 del core coincida con el esperado por el cuaderno;
3. valida el manifiesto y todos los archivos de la versión;
4. actualiza la copia activa `bundle/` dejando su manifiesto para el final;
5. extrae e instala el core en el SSD efímero `/content`;
6. exige una `NVIDIA L4` mientras `COLAB_REQUIRE_L4=True`.

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

Los checkpoints se generan en `/content` y, al activar `PUBLISH_TO_DRIVE=True`,
se publican como un TAR.GZ verificable bajo:

```text
MyDrive/ModeracionPeru_Colab/runs/<notebook_id>/<run_id>/
├── run_outputs.tar.gz
└── run_manifest.json
```

`03_01` y `03_07`–`03_08` siguen siendo adecuados para CPU local. Activar Colab
no inventa una corrida: los interruptores de entrenamiento permanecen en
`False` hasta que se complete el preflight y el smoke test.

## Secuencia práctica

1. Confirme que `resultados/colab_bundle` está actualizado; sincronícelo con
   GitHub si usará `BUNDLE_SOURCE="github"`.
2. Abra `02_00` en Google Colab, ejecute el preflight y cambie
   `RUN_PUBLISH_BUNDLE=True`.
3. Autorice `drive.mount()` y confirme `status=published_to_drive`, el
   `bundle_id` y `manifest_sha256`.
4. Abra `02_01` o `03_02`–`03_06`, seleccione Colab/L4 y ejecute desde la primera
   celda.
5. Confirme **Bundle de Colab verificado** antes de activar el procesamiento.

No hay credenciales, tokens ni IDs de carpetas personales versionados en el
proyecto.
