# Sincronización de Google Drive en `03_07a`

`03_07a_reporte_comparacion_modelos.ipynb` puede consultar directamente la
publicación de `03_07` en Google Drive desde un kernel local. No requiere Google
Drive para escritorio, no monta una unidad y no descarga modelos.

## Preparación única

1. Instale las dependencias del cuaderno desde la raíz del proyecto:

   ```powershell
   python -m pip install -e ".[cuadernos]"
   ```

2. En Google Cloud, habilite **Google Drive API** para un proyecto y cree un
   cliente OAuth de tipo **Desktop app**. Si la pantalla de consentimiento está
   en modo de prueba, agregue como usuario de prueba la cuenta que contiene la
   carpeta publicada.
3. Descargue el JSON del cliente y guárdelo como
   `config/google_drive_oauth_client.json`.
4. Ejecute la primera celda funcional de `03_07a`. Se abrirá el navegador para
   conceder únicamente el alcance `drive.readonly`.

El cliente y el token renovable `.secrets/google_drive_token.json` están
ignorados por Git. No los publique ni los comparta. Para cambiar de cuenta o
revocar la sesión local, elimine únicamente el token de `.secrets/` y vuelva a
ejecutar el cuaderno.

La sesión del conector de Google Drive disponible en Codex no se expone a un
proceso Python independiente. Por eso el cuaderno necesita esta autorización
OAuth propia una sola vez; después renueva y reutiliza el token automáticamente.

## Qué hace cada ejecución

1. Lee el puntero y los manifiestos versionados de la carpeta configurada en
   `config/google_drive_03_07.json`.
2. Selecciona la publicación válida más reciente de `03_07_working_v2_1`.
3. Compara `published_at` y el SHA-256 remoto con el estado local
   `resultados/modelos/sincronizacion_google_drive_03_07.json`.
4. Si el contenido ya coincide, no descarga el TAR. Si cambió, lo descarga o
   reconstruye sus partes y verifica tamaño y SHA-256 antes de abrirlo.
5. Extrae exclusivamente:

   - `comparacion_individual_ensemble_validation.json`
   - `seleccion_congelada.json`
   - `test_final_abierto_una_vez.json`, solo cuando exista

6. Promueve el bundle verificable más reciente a `resultados/modelos/` y
   regenera tablas, gráficas y el reporte Markdown.

Pesos, checkpoints de `Trainer` y cualquier otro miembro del TAR se ignoran. Si
todavía no existe el cliente OAuth, el cuaderno muestra una advertencia y puede
seguir produciendo el reporte a partir de los resultados locales actuales.

Referencias oficiales: [inicio rápido de Drive API para
Python](https://developers.google.com/workspace/drive/api/quickstart/python),
[búsqueda de archivos](https://developers.google.com/workspace/drive/api/guides/search-files),
[descarga de archivos](https://developers.google.com/workspace/drive/api/guides/manage-downloads)
y [alcances de autorización](https://developers.google.com/workspace/drive/api/guides/api-specific-auth).
