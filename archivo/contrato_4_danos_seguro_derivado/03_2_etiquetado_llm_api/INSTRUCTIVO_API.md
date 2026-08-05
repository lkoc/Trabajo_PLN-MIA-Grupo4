# Etiquetado mediante API — cuaderno 03.2

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Esta variante conserva el flujo del cuaderno local, pero envía los chunks a DeepSeek mediante HTTPS. Está pensada para procesar miles de muestras con bajo costo, guardado incremental y reanudación automática.

## Elección predeterminada

- Producción: `deepseek-v4-flash`, con razonamiento desactivado.
- Revisión: `deepseek-v4-pro`, aplicado solo a casos de daño, baja confianza y una muestra de control segura.
- Configuración inicial: 32 solicitudes simultáneas y 5 chunks por solicitud.
- `notes` ausente o `null` se convierte en `""`, se limita a 160 caracteres y únicamente las filas realmente inválidas se reenvían a la API.
- Si un flag transversal aparece por error dentro de `labels`, se mueve a `flags`. Así, `humor_encubridor` se conserva junto con la categoría de daño correspondiente.
- Al reanudar, las filas antiguas que ya no cumplen las reglas se respaldan, se retiran del progreso y se reetiquetan automáticamente (`QUARANTINE_INVALID_PROGRESS=True`).

Con el consumo observado en el piloto local —aproximadamente 8,28 millones de tokens de entrada y 724.000 de salida para 5.000 chunks— el costo conservador de producción con V4 Flash es cercano a **US$1,36**, suponiendo que toda la entrada sea cache miss. La caché del prefijo puede reducirlo. El precio vigente debe verificarse antes de una corrida grande en la [tabla oficial de DeepSeek](https://api-docs.deepseek.com/quick_start/pricing).

## 1. Crear y financiar la cuenta

1. Crea o abre una cuenta en [DeepSeek Platform](https://platform.deepseek.com/).
2. Genera una clave en [API Keys](https://platform.deepseek.com/api_keys).
3. Agrega un saldo pequeño. Para el piloto y 5.000 chunks, US$5 deja margen suficiente para reintentos y revisión.
4. No pegues la clave en el cuaderno ni la publiques en Git.

## 2. Configurar la credencial

Desde PowerShell, ubicado en la raíz del proyecto:

```powershell
Copy-Item '03_2_etiquetado_llm_api\.env.example' '03_2_etiquetado_llm_api\.env'
notepad '03_2_etiquetado_llm_api\.env'
```

Reemplaza el valor de ejemplo:

```dotenv
DEEPSEEK_API_KEY=sk-tu_clave_real
```

El archivo `.env` está ignorado por Git. Si prefieres no usarlo, establece la variable antes de iniciar Jupyter:

```powershell
$env:DEEPSEEK_API_KEY='sk-tu_clave_real'
jupyter lab
```

Las variables del proceso tienen prioridad sobre `.env`. Después de modificar la clave, reinicia el kernel.

## 3. Abrir y validar

Abre `03_2_etiquetado_llm_api/03_2_etiquetado_llm_api.ipynb` y ejecuta las celdas en orden. El modo inicial es:

```python
RUN_MODE = os.getenv('ETIQUETADO_RUN_MODE', 'validate').strip().lower()
```

`validate` comprueba archivos, credencial y modelos mediante `/models`; no envía textos del corpus. Debe aparecer:

```text
Credencial y API verificadas; no se enviaron textos del corpus.
```

## 4. Ejecutar el piloto

En el bloque inicial cambia el valor predeterminado a `pilot`, o define temporalmente la variable antes de iniciar Jupyter:

```powershell
$env:ETIQUETADO_RUN_MODE='pilot'
jupyter lab
```

Configuración esperada:

```python
PILOT_SAMPLE_SIZE = 300
MAX_WORKERS = 32
BATCH_SIZE = 5
```

Ejecuta todo el cuaderno y revisa:

- `chunks_per_minute`;
- `estimated_cost_usd_new_rows`;
- respuestas inválidas o HTTP 429;
- métricas contra la referencia disponible.

Si aparecen varios HTTP 429, reduce `MAX_WORKERS` primero a 16 y luego a 8. No reduzcas `BATCH_SIZE` salvo que el modelo falle al conservar el orden o cerrar el JSON.

## 5. Ejecutar 5.000 muestras

Configura al comienzo:

```python
RUN_MODE = os.getenv('ETIQUETADO_RUN_MODE', 'production').strip().lower()
PRODUCTION_SAMPLE_SIZE = 5000
SAMPLE_SEED = 42
```

También puedes usar `ETIQUETADO_RUN_MODE=production`. La selección es aleatoria y reproducible. La barra muestra chunks reales, por ejemplo `0/5000`, aunque cada solicitud agrupe cinco.

Los resultados se guardan incrementalmente en:

```text
datos/etiquetado/llm_api/deepseek-v4-flash_labeled_chunks_seed42.jsonl
```

Si se interrumpe la conexión, vuelve a ejecutar con la misma semilla, tamaño y modelo. Los `chunk_id` válidos ya guardados se omiten y solo se factura el trabajo pendiente.

## 6. Ejecutar REVIEW

Después de que exista una salida de producción, usa:

```python
RUN_MODE = os.getenv('ETIQUETADO_RUN_MODE', 'review').strip().lower()
REVIEW_SAMPLE_SIZE = 500
```

La revisión utiliza `deepseek-v4-pro`, contexto vecino y un máximo de 500 casos. No sobrescribe la primera etiqueta: crea otro JSONL. Si quieres priorizar costo sobre calidad, cambia:

```python
REVIEW_MODEL_ID = 'deepseek-v4-flash'
REVIEW_ANNOTATOR_ID = 'DSF'
```

## 7. Privacidad y control de costo

- Los textos, títulos y contexto incluidos en cada lote se envían al proveedor.
- Nunca guardes la clave dentro del notebook, JSONL o manifiesto.
- Prueba primero 50–300 chunks antes de producción.
- El costo mostrado por el cuaderno es una estimación basada en los tokens reportados por la API y la tabla incluida en la configuración.
- Si DeepSeek cambia precios, actualiza `MODEL_PRICING_USD_PER_MILLION` usando la documentación oficial.

## 8. Resolución de problemas

- `Falta DEEPSEEK_API_KEY`: crea `.env`, corrige el nombre o reinicia el kernel.
- HTTP 401/403: clave inválida, revocada o sin acceso.
- HTTP 402: saldo insuficiente.
- HTTP 429: baja `MAX_WORKERS`; el cuaderno reintenta automáticamente con espera exponencial.
- `content vacío` o JSON inválido: el cuaderno reintenta; DeepSeek advierte que JSON Output puede devolver ocasionalmente contenido vacío.
- Etiqueta fuera de la taxonomía: se conserva el resto del lote válido y se reenvía únicamente el chunk incorrecto.
- Un modelo no aparece en `/models`: revisa los IDs devueltos por el preflight y actualiza `PRIMARY_MODEL_ID` o `REVIEW_MODEL_ID`.

Documentación oficial: [JSON Output](https://api-docs.deepseek.com/guides/json_mode), [modo de razonamiento](https://api-docs.deepseek.com/guides/thinking_mode) y [límites de uso](https://api-docs.deepseek.com/quick_start/rate_limit).
