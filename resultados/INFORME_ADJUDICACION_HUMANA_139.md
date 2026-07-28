# Informe iterativo de adjudicación humana de 139 casos difíciles

Estado: **EN CURSO**  
Campaña: `revision_humana_sospechosos_139_v1`  
Última actualización: 2026-07-27T04:22:41-05:00  
Revisión acumulada del registro de progreso: 131

## Propósito

Adjudicar humanamente los 139 chunks que `deepseek-v4-pro` corrigió desde Flash-seguro a daño, pero para los cuales conservó `needs_review=True`. El objetivo humano se limita a `SEGURO` o una/más de cinco categorías gruesas de daño. Las etiquetas finas de Pro se conservan solo como referencia y no se entrenan.

## Embudo cuantitativo Flash → Pro → humano

La ruta operativa tiene dos detectores de duda distintos. Primero, Flash expresa duda mediante el booleano `needs_review` o mediante su autoevaluación `score_confianza`. La regla congelada es `needs_review_flash OR score_confianza < 0.90`. `score_confianza` no es un intervalo estadístico ni se interpreta literalmente como probabilidad; su utilidad se midió empíricamente por acuerdo con Pro. Segundo, entre los seguros que Flash consideró confiables, el moderador grueso calculó `score_dano_maximo` y envió a Pro los 2.000 valores más altos de train+validation como auditoría dirigida. Finalmente, Pro expresa duda persistente con `needs_review=True`; por contrato, cualquier flag transversal o `score_confianza < 0.70` obliga esa marca, aunque Pro puede activarla adicionalmente.

| Etapa | Chunks | % del corpus Flash | % del padre relevante |
|---|---:|---:|---:|
| Corpus etiquetado por Flash | 69,853 | 100,000% | 100,000% |
| Flash asigna una categoría segura | 64,299 | 92.049% | 92.049% del total |
| Duda auto-reportada por Flash: alerta o score < 0,90 | 6,116 | 8.756% | 8.756% del total |
| Seguros Flash dentro de esa duda auto-reportada | 563 | 0.806% | 0.876% de seguros Flash |
| Seguros Flash aceptados por la regla de confianza | 63,736 | 91.243% | 99.124% de seguros Flash |
| Revisiones Pro previas (dudas + daño + controles) | 11,421 | 16.350% | cubren 6,116/6,116 dudas Flash |
| Seguros confiables seleccionados por score de daño para Pro | 2,000 | 2.863% | 3.138% de seguros confiables |
| Pro corrige esos seguros a daño | 245 | 0.351% | 12.250% de los 2.000 |
| Duda persiste en Pro y pasa a humano | 139 | 0.199% | 56.735% de correcciones Pro; 6.950% de los 2.000 |
| Intervención humana requerida | 139 | 0.199% | 1.036% de todos los chunks vistos por Pro |

Después de la minería dirigida, Pro ha visto 13,421 IDs únicos (19.213% del corpus), mientras que la intervención humana se concentra en 139 (0.199% del corpus). La reducción Pro→humano es, por tanto, de 98.964% respecto de todo lo revisado por Pro.

### Umbral y métrica de duda de Flash

- Regla operativa: `needs_review_flash OR score_confianza < 0.90`.
- Tasa de derivación de esa regla: 8.756%; cobertura automática: 91.244%.
- Entre los casos cubiertos automáticamente, acuerdo exacto Flash–Pro: 93.192%, con límite inferior unilateral de 95% de 92.530%.
- Acuerdo binario daño/seguro: 96.548%, con límite inferior unilateral de 95% de 96.037%.
- Captura de desacuerdos: 47.184% para etiqueta exacta y 52.840% para daño/seguro.
- Estas métricas miden consistencia con Pro en el diseño de validación; no convierten a Pro en verdad humana.

### Métrica de duda persistente de Pro

De los 139 casos enviados a humano, 133 contienen al menos un flag transversal y 121 tienen `score_confianza < 0.70`; 121 cumplen ambas condiciones. Otros 6 fueron marcados explícitamente por Pro aun sin flag ni score inferior a 0,70.
Los flags acumulados son: humor encubridor=93, contexto necesario=40 e ironía ambigua=11. Puede existir más de un flag por chunk.

## Diseño y controles

- Universo de la campaña: los 139 casos Pro con duda residual dentro de la revisión dirigida de 2.000.
- Partición original: 114 train y 25 validation; cero casos y cero videos de test.
- Orden: barajado determinísticamente con semilla 13942.
- Decisión humana primaria: categorías gruesas multilabel; `SEGURO` es excluyente con cualquier daño.
- Flags transversales: ironía ambigua, humor encubridor y contexto necesario; requieren una categoría de daño y nunca se convierten en categorías base.
- Reducción de anclaje: la sugerencia Pro permanece oculta hasta que el anotador elige al menos una categoría; la revelación queda registrada.
- Trazabilidad: cada guardado incrementa una revisión, actualiza un snapshot atómico y añade un evento JSONL inmutable.
- Cierre: el JSONL final solo se escribe cuando los 139 casos tienen `status=completed`; los borradores o diferidos bloquean la integración.

Este procedimiento aplica revisión dirigida de posibles errores de etiqueta y documenta la decisión humana independiente antes de consultar la sugerencia del modelo (Brodley & Friedl, 1999; Settles, 2009). Al existir un solo adjudicador, no es posible estimar acuerdo interanotador; esta limitación debe declararse en el paper (Artstein & Poesio, 2008).

## Progreso

| Estado | n |
|---|---:|
| Completados | 26 |
| Diferidos | 0 |
| Borradores | 0 |
| Sin abrir | 113 |
| Pendientes para producir el set final | 113 |

## Resultados humanos acumulados

| Categoría gruesa | Positivos completados |
|---|---:|
| `SEGURO` | 6 |
| `RACISMO_DISCRIMINACION` | 8 |
| `ACOSO_GENERO_IDENTIDAD` | 6 |
| `ACOSO_PERSONAL` | 7 |
| `AMENAZA_DIRECTA` | 1 |
| `CONTENIDO_SEXUAL` | 8 |

- Decisiones completas disponibles: 26.
- Acuerdo exacto acumulado humano–Pro: 15/26
- Humano asigna SEGURO: 6.
- Humano asigna daño: 20.

Estas cifras son descriptivas del estrato difícil y no estiman el desempeño poblacional de Flash o Pro.

## Integración prevista al entrenamiento

Cuando la campaña esté completa y sea válida, el pipeline aplicará la precedencia `humano grueso > Pro > Flash` para estos IDs, conservará su split original y entrenará exclusivamente las seis categorías de `COARSE_ORDER`. Antes de completarse, el flujo de reentrenamiento debe permanecer bloqueado.

## Artefactos

- Campaña: `datos\etiquetado\humano\revision_humana_sospechosos_139.campaign.json`
- Manifiesto: `datos\etiquetado\humano\revision_humana_sospechosos_139.campaign.manifest.json`
- Progreso atómico: `datos\etiquetado\humano\revision_humana_sospechosos_139.progress.json`
- Eventos iterativos: `datos\etiquetado\humano\revision_humana_sospechosos_139.events.jsonl`
- Salida final: `datos\etiquetado\humano\revision_humana_sospechosos_139.jsonl`
- Frontend: `Cuadernos\frontend\revision_humana_sospechosos_139.html`
- SHA-256 canónico: `eb90debf66d5e16af72c41c17c3701197e42bdcc78b81e0f914c6a49c56f8ab4`
- SHA-256 Pro-2000: `745a9dc191482fd0a8609f5ffec3266abcf7a0bd40860d7c7b541f7dac045e34`
- SHA-256 calibración de confianza: `99a28c17a90aa676915811013aac9dcbae7e359d3783fe1e78c1935b0f1e09bf`
- SHA-256 campaña: `deb838cd5553142666a4dc0cd8b6b185ddf6aecff1323a38b638475b3b14f329`
- SHA-256 salida final: `pendiente`

## Limitaciones

- La campaña es deliberadamente dirigida y no probabilística.
- Participa un solo adjudicador; no se puede calcular kappa/alpha interanotador.
- Revelar Pro después de la primera decisión puede aún influir en revisiones posteriores; el campo `pro_revealed` permite auditarlo.
- Los casos diferidos no deben entrar al entrenamiento hasta resolverse.

## Referencias

Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics, 34*(4), 555–596. https://doi.org/10.1162/coli.07-034-R2

Brodley, C. E., & Friedl, M. A. (1999). Identifying mislabeled training data. *Journal of Artificial Intelligence Research, 11*, 131–167. https://doi.org/10.1613/jair.606

Settles, B. (2009). *Active learning literature survey* (Computer Sciences Technical Report 1648). University of Wisconsin–Madison. https://research.cs.wisc.edu/techreports/2009/TR1648.pdf
