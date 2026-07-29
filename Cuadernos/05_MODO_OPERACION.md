# Modo de operación del moderador local

## Alcance

`05_frontend_produccion.ipynb` inicia una aplicación local para moderación asistida. La página es un único archivo HTML con CSS, JavaScript y ayuda embebidos; los checkpoints se ejecutan en un backend Python local y no se insertan dentro del navegador. El servidor escucha en `127.0.0.1` de forma predeterminada.

El sistema no autoriza decisiones autónomas de bloqueo o sanción. “Confianza alta” significa distancia respecto de umbrales ajustados en validation, no certeza de que la clasificación sea correcta.

## Inicio rápido

1. Ejecute `04_207` y `04_208` para actualizar comparación, registro desplegable y auditoría.
2. Abra `05_frontend_produccion.ipynb` desde la raíz o desde `Cuadernos/`.
3. Ejecute las celdas de dependencias y registro de modelos.
4. Ejecute “Crear `05_frontend_despliegue`” para producir el paquete portable.
5. Ejecute la celda “Iniciar la página HTML local”.
6. Abra `http://127.0.0.1:8765/` si el navegador no se abre automáticamente.

La primera inferencia de Transformer o Qwen puede tardar porque el modelo se carga bajo demanda. Las llamadas posteriores reutilizan el modelo en memoria.

## Entrada minimalista y detección automática

La pantalla principal tiene una sola caja:

- un enlace válido de `youtube.com` o `youtu.be` activa el modo YouTube;
- cualquier otra entrada se trata como texto;
- “Ajustes” permite forzar el tipo sólo para depuración.

En YouTube se aceptan subtítulos manuales o automáticos. Si no existe ninguna pista descargable, la solicitud se rechaza; el servidor no descarga ni transcribe el audio. Los segmentos se convierten en chunks cercanos a 30 segundos o 600 caracteres, igual que el pipeline de preentrenamiento. Cada chunk con alerta incluye un enlace al segundo inicial y la página muestra el reproductor del video.

## Selección visible en la pantalla principal

El selector principal ofrece:

- `Mejor ML clásico`: SVM plano seleccionado por PR-AUC de validation.
- `Mejor MiniLM Transformer`: E5-small de linaje MiniLM, ganador de `04_202` por validation.
- `Qwen fine-tuned`: época operativa 3, elegida antes de test al mismo objetivo de recall.
- `Comparar las tres respuestas`: muestra las salidas por separado.
- `Consenso de los tres`: activa una categoría con al menos dos votos. Es una
  mayoría 2 de 3; no requiere unanimidad.

El registro verificable está en `resultados/metricas/comparacion_final_4/registro_modelos_desplegables.json`. Contiene rutas, SHA-256, thresholds, regla de selección y política de revisión. `05` se detiene si un artefacto cambió.

## Confianza y revisión humana

Para cada modelo se calculan dos señales fijadas sólo con validation:

1. una ruta de alto recall basada en `max(score - threshold)`, ajustada al objetivo de 95 %;
2. cercanía a los umbrales, usando el 20 % de márgenes más pequeños de validation.

La página muestra “Revisión necesaria” si se activa cualquiera. En consenso también se exige revisión cuando los modelos discrepan. Las acciones son:

- `Aceptar`: conserva las etiquetas propuestas;
- `Rechazar`: adjudica `SEGURO`;
- `Modificar`: permite escoger manualmente uno o varios daños.

El nombre o iniciales del revisor y una nota son opcionales. Un evento no puede revisarse dos veces; una corrección posterior debe registrarse mediante un proceso de adjudicación explícito, no sobrescribiendo el historial.

## Estadísticas y persistencia

El backend guarda:

- `resultados/operacion_05/estadisticas_moderacion.sqlite3`: inferencias, scores, modelo, categoría, confianza y acciones humanas;
- `resultados/operacion_05/revisiones_para_reentrenamiento.jsonl`: bitácora append-only de todas las revisiones;
- `resultados/operacion_05/revisiones_adjudicadas_unicas.jsonl`: dataset deduplicado; decisiones contradictorias sobre el mismo texto se excluyen.

La pantalla “Estadísticas” separa conteos por modelo y categoría, revisiones solicitadas/completadas y acciones aceptar/rechazar/modificar.

La señal orientativa de suficiencia para revisar un reentrenamiento exige 500 chunks humanos únicos, 200 seguros y 100 positivos por cada daño. Alcanzarla no inicia un entrenamiento automático. Primero deben revisarse conflictos, diversidad de videos, representatividad temporal, privacidad y distribución; luego se crea un nuevo train y un nuevo test prospectivo. Los ejemplos de producción nunca se agregan al validation/test histórico.

## Ejecución directa desde el cuaderno

Una sola celda contiene los parámetros equivalentes a la página:

```python
ENTRADA = "una frase o enlace"
TIPO_ENTRADA = "auto"
MODO_MODELO = "consensus"
IDIOMAS_SUBTITULOS = ("es", "es-419", "es-US", "en")
MAX_CHUNKS = 300
GUARDAR_ESTADISTICAS = True
```

`MODO_MODELO` admite `classical`, `transformer`, `qwen`, `compare` y `consensus`.

## Carpeta portable `05_frontend_despliegue`

El cuaderno genera una carpeta independiente de aproximadamente 1.7 GiB con:

- HTML, backend Python y ayuda;
- SVM, E5-small, Qwen base, adaptador LoRA y calibradores;
- registro de selección y hashes de integridad;
- una SQLite nueva y JSONL vacíos, sin copiar estadísticas privadas anteriores;
- `app.py`, `requirements.txt`, `Dockerfile` y `docker-compose.yml`;
- `requirements-lock.txt` y `build_environment.json` con versiones exactas;
- `README.md` y `GUIA_DESPLIEGUE.md` para ejecución local o publicación como
  Docker Space de Hugging Face.

La carpeta queda ignorada por el repositorio principal debido al tamaño de los
pesos. Puede cargarse como repositorio de despliegue independiente; incluye
reglas Git LFS para los archivos grandes.

El empaquetador exige que la auditoría de 04_208 tenga el mismo dataset y la
misma selección Qwen, que incluya los tres modelos publicados y que no declare
resultados pendientes. Por ello, la secuencia de cierre es `04_207 → 04_208 →
05`; después de esos dos cuadernos se reconstruye el bundle.

Para acceso externo se admite autenticación HTTP Basic mediante
`MODERATOR_ACCESS_USER` y `MODERATOR_ACCESS_PASSWORD`. Debe colocarse detrás de
HTTPS. La SQLite sólo persiste si `data/` reside en disco o volumen persistente.

La reconstrucción es funcionalmente reproducible y verificable por hashes. No
es idéntica byte por byte: cambian la fecha del manifiesto, la ruta de origen y
metadatos internos de la SQLite vacía.

## Seguridad y límites

- No exponga el servidor del cuaderno directamente. Use el bundle, contraseña,
  HTTPS y un volumen persistente para acceso web externo.
- Sólo se aceptan hosts de YouTube conocidos; no se descargan URLs arbitrarias.
- El HTML no usa bibliotecas, fuentes ni scripts de terceros. El iframe de YouTube es el único recurso web visual.
- Los textos y revisiones quedan en disco local y pueden contener información sensible. Aplique la política de retención correspondiente.
- `yt-dlp` depende de cambios de YouTube; actualícelo si la extracción falla, sin cambiar retrospectivamente los modelos.
- Para construir el bundle, Qwen base debe estar en caché o poder descargarse.
  Después de empaquetarlo, la inferencia usa su copia local y funciona offline.

## Endpoints locales

- `GET /api/health`: estado del servidor.
- `GET /api/config`: modelos y configuración pública.
- `GET /api/stats`: estadísticas y suficiencia orientativa.
- `POST /api/analyze`: análisis de texto o YouTube.
- `POST /api/review`: aceptación, rechazo o modificación humana.

La ayuda resumida de esta guía está incluida dentro del botón “Ayuda” del HTML.
