# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 04 · Producción supervisada

El frontend opera en modo sombra con el contrato v2.1: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. No bloquea, elimina ni sanciona contenido; presenta scores, umbrales, motivos de revisión y conserva la decisión final en una persona supervisora.

## Modelos vigentes

El registro productivo debe provenir de `resultados/modelos/seleccion_congelada.json`. La selección actual contiene cuatro sistemas visibles:

1. Clásico: `classical-logistic_regression_c0p5-54f7971c6000`.
2. Transformer: `cascade_v2-af78eba77883`.
3. Qwen: `qwen_lora-4aa5ce04df05`.
4. Ensemble ganador: `ensemble_soft_mean`.

El ensemble promedia los scores crudos de los tres miembros, aplica los cinco calibradores sigmoidales congelados y luego los umbrales de despliegue. La regla `NEEDS_REVIEW` reproduce el margen seleccionado en validation y revisa conflictos `SEGURO`--daño, salida vacía, incoherencia con la compuerta binaria y cercanía a los umbrales. El estado `statistical_tie_or_inconclusive` se conserva en el registro: el promedio suave ocupa el primer lugar por el criterio predeclarado, pero no se presenta como superioridad estadística demostrada.

## Preparación del registro

Los pesos no se versionan en Git. Una vez restaurados los tres directorios de candidato desde Google Drive, materialice el registro con la función versionada:

```python
from moderacion_peru.registry import publish_frozen_ensemble_registry

publish_frozen_ensemble_registry(
    "resultados/modelos/seleccion_congelada.json",
    ["modelos/v2"],
    "modelos/registro_modelos_5_salidas.json",
)
```

La función verifica los identificadores, el SHA-256 del snapshot, los manifiestos de checkpoint y cada archivo declarado. Genera un registro principal y tres registros miembro. Si falta un candidato, se detiene de forma explícita y nunca reemplaza el ganador por una versión histórica.

## Inicio local

```powershell
modperu serve-production --host 127.0.0.1 --port 8765
```

La interfaz acepta texto o una URL de YouTube, usa únicamente subtítulos, conserva tiempos por fragmento y permite ejecutar el ensemble, cada miembro por separado o la comparación de los cuatro sistemas. Las inferencias y revisiones se guardan de forma append-only; las estadísticas se separan por modelo y categoría. La retroalimentación concordante se deduplica en `datos/produccion/retraining_ready_v2.jsonl` y los conflictos se excluyen.

Por defecto el servidor solo escucha en loopback. Para exponerlo deliberadamente en otra interfaz se requiere `MODERATOR_ACCESS_PASSWORD` y, opcionalmente, `MODERATOR_ACCESS_USER`.
