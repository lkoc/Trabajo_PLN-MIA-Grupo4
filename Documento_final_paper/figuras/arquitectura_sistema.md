# Arquitectura operacional del demostrador

Este diagrama resume el flujo verificable del moderador textual local. Una sola caja acepta una frase o una URL: la interfaz detecta el tipo de entrada y rechaza enlaces de YouTube sin subtítulos descargables. La versión de adquisición y evaluación usada en el paper está en `pipeline_moderacion.tex`; la arquitectura del demostrador está en `despliegue.tex`.

```mermaid
flowchart LR
  A[Texto o URL de YouTube] --> B{Deteccion automatica}
  B -->|Frase| C[Normalizacion y chunk unico]
  B -->|URL con subtitulos| D[Descarga de subtitulos y tiempos]
  B -->|URL sin subtitulos| X[Rechazo explicito]
  D --> E[Chunks equivalentes al entrenamiento]
  C --> F{Modo de inferencia}
  E --> F
  F --> G[SVM clasica]
  F --> H[E5 Transformer]
  F --> I[Qwen ajustado]
  G --> J[Modelo unico, comparacion o consenso]
  H --> J
  I --> J
  J --> K[Alerta, confianza y evidencia temporal]
  K --> L[Decision humana: aceptar, modificar o rechazar]
  L --> M[Base de datos y registro versionado para estadística y reentrenamiento]
```

El consenso es una mayoría de al menos dos de los tres modelos, no unanimidad. El sistema materializa moderación semiautomática: prioriza, enlaza el diagnóstico con tiempos del video y permite que el supervisor acepte, modifique o rechace. Una alerta informa esa decisión y no constituye por sí sola una sanción. No se afirma uso de reconocimiento automático del habla; el modo URL exige una pista de subtítulos disponible.
