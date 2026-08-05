# Informe de adjudicación humana combinada: 3,114 casos

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Estado: **COMPLETADA**  
Campaña: `revision_humana_combinada_1918_v2`  
Actualización: 2026-07-27T09:54:14-05:00  

## Alcance y procedencia

La campaña integra de forma append-only la cola original y 3 colas de ampliación descubiertas automáticamente. Los manifiestos verifican procedencia, orden y hashes; ningún caso de ampliación pertenece a test. Total actual: **3,114**.

| Cohorte | Total | Resueltos | Faltan | Incluidos | Excluidos |
|---|---:|---:|---:|---:|---:|
| Corrida original (139) | 139 | 139 | 0 | 135 | 4 |
| Segunda corrida / ampliación (1.779) | 1,779 | 1,779 | 0 | 1,775 | 4 |
| Ampliación ampliacion_dano_20260727_lote2 (1,088) | 1,088 | 1,088 | 0 | 1,074 | 14 |
| Ampliación ampliacion_amenaza_20260727_lote3 (108) | 108 | 108 | 0 | 108 | 0 |
| **Total** | **3,114** | **3,114** | **0** | **3,092** | **22** |

## Regla operativa y trazabilidad

La propuesta de `deepseek-v4-pro` se presenta antes de decidir para acelerar la adjudicación. Cada clic queda registrado con fecha, iniciales, cohorte, revisión y acción:

- `accept_llm`: copia en servidor las categorías gruesas y flags de Pro; `training_eligible=true`.
- `reject_llm`: vacía categorías y flags; `training_eligible=false`; el chunk queda fuera del entrenamiento.
- `modify_llm`: conserva la versión gruesa elegida por el humano y sus flags; `training_eligible=true`.
- `legacy_human_decision`: preserva como tal una decisión realizada antes de introducir los botones rápidos; no se afirma retrospectivamente que fue un clic de aceptación.

Las etiquetas finas de Pro se muestran solo como contexto y nunca son objetivos de entrenamiento. Los flags transversales se mantienen separados de las categorías base.

## Estado de las acciones

- Aceptaciones explícitas: 1,604.
- Rechazos/exclusiones: 22.
- Modificaciones humanas: 1,462.
- Decisiones humanas migradas: 26.
- Borradores: 0; diferidos: 0; sin abrir: 0.

## Integración con entrenamiento

Los archivos finales incluyen todas las decisiones para conservar la auditoría. Los pipelines filtran explícitamente `training_eligible=false`: aceptar usa la versión Pro, modificar usa la versión humana y rechazar elimina el chunk. Cada cohorte se materializa cuando termina; la salida combinada se genera al completar las 3,114 decisiones vigentes.

## Artefactos reproducibles

- Campaña: `datos\etiquetado\humano\revision_humana_combinada_1918.campaign.json`.
- Manifiesto: `datos\etiquetado\humano\revision_humana_combinada_1918.campaign.manifest.json`.
- Progreso: `datos\etiquetado\humano\revision_humana_combinada_1918.progress.json`.
- Eventos: `datos\etiquetado\humano\revision_humana_combinada_1918.events.jsonl`.
- Salida original: `datos\etiquetado\humano\revision_humana_sospechosos_139.jsonl`.
- Salida combinada: `datos\etiquetado\humano\revision_humana_combinada_1918.jsonl`.
- Frontend: `Cuadernos\frontend\revision_humana_sospechosos_139.html`.
- SHA-256 cola original Pro: `745a9dc191482fd0a8609f5ffec3266abcf7a0bd40860d7c7b541f7dac045e34`.
- Cola `ampliacion_dano_20260726`: `datos\ampliacion\ampliacion_dano_20260726\processed\pendientes_revision_humana.jsonl`; SHA-256 `d65622af30683f2821ac89629055222189b4d0597a9346689d6c31429d35d305`.
- Cola `ampliacion_dano_20260727_lote2`: `datos\ampliacion\ampliacion_dano_20260727_lote2\processed\pendientes_revision_humana.jsonl`; SHA-256 `56430b83aa953ddabfc050eac581e29dcadb03d26e9fc7642bf53f646bb63c1d`.
- Cola `ampliacion_amenaza_20260727_lote3`: `datos\ampliacion\ampliacion_amenaza_20260727_lote3\processed\pendientes_revision_humana.jsonl`; SHA-256 `c8edd437c206a2f5eb985a8f8c7bc51e927437f6aa777b8578ab27a9a5f134cb`.

Inicio desde la raíz:

```powershell
python -m scripts_auxiliares.servidor_revision_humana_139 --host 127.0.0.1 --port 8765
```

## Limitación metodológica

Mostrar la propuesta antes de la decisión aumenta velocidad, pero puede producir anclaje. Por ello estas decisiones sirven para depurar y construir entrenamiento, no para estimar de manera ciega e independiente el error de Pro. Una medición de acuerdo humano–LLM requiere una submuestra ciega o doble anotación independiente.

## Referencias (APA 7)

Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics, 34*(4), 555–596. https://doi.org/10.1162/coli.07-034-R2

Brodley, C. E., & Friedl, M. A. (1999). Identifying mislabeled training data. *Journal of Artificial Intelligence Research, 11*, 131–167. https://doi.org/10.1613/jair.606

Settles, B. (2009). *Active learning literature survey* (Computer Sciences Technical Report 1648). University of Wisconsin–Madison. https://research.cs.wisc.edu/techreports/2009/TR1648.pdf
