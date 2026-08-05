# Plan de reorganización reproducible

Estado: aprobado para implementación  
Fecha: 2026-08-05  
Proyecto: moderación semiautomática de contenido peruano de YouTube

## Objetivo

Reorganizar el proyecto en cuatro etapas ejecutables, mantener la evidencia ya producida y establecer desde el etiquetado un contrato de cinco salidas entrenadas:

1. `SEGURO`;
2. `RACISMO_DISCRIMINACION`;
3. `ACOSO_GENERO_IDENTIDAD`;
4. `ACOSO_AMENAZA`;
5. `CONTENIDO_SEXUAL`.

`SEGURO` será una categoría supervisada, con score, umbral y métricas propios, y será mutuamente excluyente con cualquier daño. Los cuatro daños seguirán siendo multietiqueta. Se conservarán las 14 etiquetas finas y los tres flags transversales.

Los resultados actuales de cuatro daños con `SEGURO` derivado se preservarán como línea base ejecutada. No se les atribuirá validez bajo el nuevo contrato sin reentrenamiento.

## Estructura objetivo

```text
README.md
Planning/
config/
docs/
src/moderacion_peru/
flujo/
├── 01_datos/
├── 02_etiquetado/
├── 03_entrenamiento/
└── 04_produccion/
datos/
modelos/
resultados/
archivo/
Documento_final_paper/
Presentación_BEAMER/
bibliografia/
Guias_generales/
```

### Orden de cuadernos

| Etapa | Cuadernos activos |
|---|---|
| Datos | `01_01_scraping_incremental`, `01_02_limpieza_troceado_incremental`, `01_03_ampliacion_dirigida` |
| Etiquetado | `02_01_etiquetado_local_ollama`, `02_02_etiquetado_remoto`, `02_03_revision_llm_dirigida`, `02_04_consolidacion_validacion_humana` |
| Entrenamiento | `03_01` clásicos, `03_02` Transformers planos, `03_03` cascada, `03_04` multitarea, `03_05` Qwen-LoRA, `03_06` Qwen estructurado, `03_07` comparación y `03_08` auditoría |
| Producción | `04_01_frontend_produccion` |

## Implementación

### Preservación y archivo

- Generar inventario SHA-256 previo a la reorganización.
- Archivar los contratos anteriores, LM Studio, cuadernos duplicados y material de clase sin borrar evidencia.
- Mantener intactas las salidas ejecutadas; las nuevas corridas escribirán snapshots versionados.
- Eliminar rutas personales y resolver la raíz mediante `pyproject.toml`, `MODPERU_ROOT` y `MODPERU_ARTIFACT_ROOT`.

### Núcleo común

- Crear el paquete `moderacion_peru` con rutas, contratos Pydantic, taxonomía, manifiestos, E/S incremental, selección de dispositivo y proveedores de LLM.
- Exponer el CLI `modperu`: `preflight`, `run-stage`, `validate`, `serve-labeling`, `serve-production` y `artifacts`.
- Admitir `auto`, NVIDIA/CUDA, AMD/ROCm, Intel/XPU y CPU, con registro explícito del backend y fallback advertido.

### Etiquetado y taxonomía

- Crear `taxonomia_v2` con definición, inclusión, exclusión, contraejemplos, mapeos finos y fuentes.
- Corregir ambigüedades sobre amenaza, acoso, discurso reportado, humor, ironía y contexto.
- Usar `needs_review=true`, `training_eligible=false` y categorías vacías para casos indeterminados.
- Migrar sin sobrescribir: `ACOSO_PERSONAL ∪ AMENAZA_DIRECTA → ACOSO_AMENAZA`; `SEGURO` solo desde decisiones seguras explícitas.
- Reemplazar LM Studio por Ollama HTTP con JSON Schema; conservar DeepSeek remoto y Hugging Face local como adaptadores opcionales.

### Datos y entrenamiento incremental

- Mantener IDs deterministas, particiones agrupadas por video y manifiestos con hashes.
- Procesar videos/chunks nuevos por diferencia y reanudar por `chunk_id`.
- Reutilizar encoders o backbones compatibles, crear una nueva cabeza de cinco salidas y recalibrar umbrales.
- Evaluar las cinco categorías, los cuatro daños, falsos seguros, conflictos, calibración y carga de revisión.

### Frontends

- Servir campañas humanas desde archivos externos, sin datos masivos incrustados en HTML.
- Impedir `SEGURO` junto con daño y admitir diferir/abstenerse.
- Mostrar en producción cinco scores aprendidos, contrato, modelo, evidencia temporal y motivos de revisión.
- Mantener el sistema como demostrador local o modo sombra, no como moderación autónoma.

### Documentación académica

- Crear un README raíz breve y READMEs por etapa.
- Auditar todos los Markdown, corregir rutas y añadir estado/contrato a documentos históricos.
- Crear matriz de trazabilidad de afirmaciones, fuentes y artefactos.
- Conservar la taxonomía v1.3 y producir v2 sin sobrescribirla.
- Preservar paper y Beamer actuales por hash; actualizar su narrativa al nuevo diseño y mantener sus métricas como línea base del contrato anterior.

## Validación

- Pruebas de esquema, mapeo, exclusividad, precedencia y migración.
- Pruebas de idempotencia y reanudación.
- Smoke tests offline de datos, Ollama, entrenamiento y frontends.
- Verificación estructural de notebooks con `nbformat`.
- Pruebas de selección CUDA/ROCm/XPU/CPU.
- Auditoría de enlaces, rutas, manifiestos y hashes.
- Compilación limpia de paper y Beamer y revisión de citas.

No se realizarán llamadas comerciales, etiquetado completo ni reentrenamiento integral durante esta implementación. El piloto local de Ollama se ejecutará solo si los modelos requeridos están disponibles o pueden descargarse de forma razonable; de otro modo quedará un diagnóstico reproducible y una orden explícita de ejecución.
