# Informe del primer intento de entrenamiento con etiquetas gruesas

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


> **Informe histórico del contrato de cinco etiquetas.** No corresponde a los cuadernos activos `04_201`–`04_208`.

**Fecha de ejecución:** 26 de julio de 2026  
**Cuaderno:** `Cuadernos/04_entrenamiento_moderador.ipynb`  
**Estado del experimento:** baseline reproducido de extremo a extremo y congelado antes de aplicar mejoras.

## 1. Objetivo

El experimento comparó clasificadores locales para asignar cinco categorías gruesas de daño o `SEGURO` a chunks de transcripciones de videos peruanos de YouTube:

1. `RACISMO_DISCRIMINACION`;
2. `ACOSO_GENERO_IDENTIDAD`;
3. `ACOSO_PERSONAL`;
4. `AMENAZA_DIRECTA`;
5. `CONTENIDO_SEXUAL`;
6. `SEGURO`, únicamente cuando no se predice alguna de las cinco categorías de daño.

Las 14 etiquetas finas se usaron para construir determinísticamente estos objetivos y para auditar las particiones, pero no se incorporaron como variables predictoras ni se entrenaron como salidas. Los flags `ironia_ambigua`, `humor_encubridor` y `contexto_necesario` se reservaron para calibrar la derivación a revisión.

## 2. Datos y procedencia de las etiquetas

El corpus híbrido contiene 69,853 chunks:

| Fuente de la etiqueta final | Chunks |
|---|---:|
| DeepSeek Pro | 11,421 |
| DeepSeek Flash como pseudoetiqueta | 58,432 |

Pro reemplaza a Flash siempre que existe revisión. Las pseudoetiquetas Flash reciben un peso de entrenamiento de `0.50 × score_confianza`; las de Pro reciben peso 1. Todos los ejemplos que permanecieron exclusivamente con Flash fueron clasificados como seguros. Todos los ejemplos de daño del dataset híbrido proceden de Pro, lo que debe considerarse al interpretar el posible sesgo de selección.

La distribución gruesa antes de dividir fue aproximadamente 95.1% `SEGURO`. Los recuentos positivos de daño fueron:

| Categoría | Positivos |
|---|---:|
| Racismo/discriminación | 1,030 |
| Acoso por género/identidad | 963 |
| Acoso personal | 1,156 |
| Amenaza directa | 210 |
| Contenido sexual | 901 |

Las categorías de daño no son mutuamente excluyentes, por lo que sus recuentos pueden solaparse.

## 3. Separación experimental

Se evaluaron 250 divisiones candidatas y se escogió la de semilla 131, minimizando diferencias de prevalencia sin permitir que un mismo `video_id` apareciera en más de una partición.

| Partición | Chunks | Videos | Uso |
|---|---:|---:|---|
| Entrenamiento | 48,927 | 1,297 | Ajuste de parámetros |
| Validación | 10,633 | 279 | Umbrales y selección del modelo |
| Prueba | 10,293 | 279 | Evaluación posterior a la selección |

No hubo videos compartidos entre entrenamiento, validación y prueba. Después de evaluar, el artefacto de producción fue reajustado con entrenamiento + validación (59,560 chunks), sin incorporar la prueba. A la fecha del experimento no existía un conjunto de consenso humano; por tanto, la prueba usa etiquetas híbridas Pro/Flash y no constituye validación contra un estándar humano.

## 4. Modelos comparados

Se compararon cinco modelos con la misma partición y los mismos pesos por fuente:

1. Dummy por prevalencia, que ignora el texto.
2. Complement Naive Bayes con TF-IDF.
3. Regresión logística con TF-IDF de unigramas y bigramas.
4. SVM lineal con n-gramas de palabras y caracteres.
5. Gradient Boosting sobre TF-IDF reducido mediante SVD.

Los umbrales se ajustaron en validación maximizando F1 por salida. El ganador se seleccionó por F1 macro calculado exclusivamente sobre las cinco categorías de daño, con PR-AUC macro de daño como desempate. `SEGURO` no participó en la métrica primaria de selección.

## 5. Resultados

| Modelo | F1 macro daño, validación | F1 macro daño, prueba | Recall micro daño | PR-AUC macro daño | Exact match | Entrenamiento (s) | Inferencia (ms/1,000) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SVM palabra + carácter** | **0.2424** | **0.2752** | 0.2787 | **0.2317** | 0.9226 | 57.3 | 410.1 |
| Regresión logística | 0.2257 | 0.2659 | **0.3071** | 0.2101 | 0.9252 | 9.3 | 90.5 |
| Gradient Boosting + SVD | 0.1068 | 0.1077 | 0.2929 | 0.0699 | 0.8320 | 10.5 | 90.0 |
| Complement Naive Bayes | 0.0683 | 0.0500 | 0.0898 | 0.0322 | 0.9071 | 2.8 | 57.3 |
| Dummy | 0.0000 | 0.0000 | 0.0000 | 0.0123 | **0.9520** | <0.1 | 0.2 |

La exactitud elevada del baseline Dummy demuestra que `exact match` no es una métrica suficiente: responder siempre `SEGURO` alcanza 95.2% debido al desbalance, aunque no detecta ningún daño.

### 5.1 Desempeño del ganador por categoría

| Categoría de daño | Precisión | Recall | F1 | Soporte de prueba |
|---|---:|---:|---:|---:|
| Racismo/discriminación | 0.3933 | 0.2482 | 0.3043 | 141 |
| Acoso por género/identidad | 0.3067 | 0.3650 | 0.3333 | 137 |
| Acoso personal | 0.2085 | 0.2444 | 0.2251 | 180 |
| Amenaza directa | 0.1141 | 0.3091 | 0.1667 | 55 |
| Contenido sexual | 0.5439 | 0.2541 | 0.3464 | 122 |

La SVM convergió holgadamente: los seis estimadores usaron entre 22 y 270 iteraciones frente a un máximo permitido de 5,000. El desempeño bajo no se atribuye a falta de tiempo de optimización.

## 6. Enrutamiento de casos dudosos

El margen de incertidumbre se calibró en validación para capturar al menos 80% de los chunks con algún flag transversal. En prueba:

| Indicador | Resultado |
|---|---:|
| Proporción derivada a revisión | 39.8% |
| Cobertura automática | 60.2% |
| Flags capturados | 82.0% |
| Proporción de revisados que tenía flag | 6.6% |

La captura objetivo se logra a costa de revisar aproximadamente cuatro de cada diez chunks.

## 7. Conclusión del primer intento

La SVM de palabras y caracteres fue el mejor modelo según el criterio declarado, pero su F1 macro de daño de 0.275 y recalls entre 0.244 y 0.365 son insuficientes para moderación autónoma. El artefacto puede utilizarse únicamente como priorizador experimental de revisión humana. No debe aprobar, bloquear, sancionar ni publicar contenido sin supervisión.

La afirmación se limita además porque todavía no existe un holdout humano independiente. Este resultado constituye el baseline contra el que deben compararse las mejoras posteriores; no debe sobrescribirse ni presentarse como el resultado definitivo del proyecto.

## 8. Artefactos reproducibles

- Métricas: `resultados/metricas/moderador_grueso/comparacion_modelos_completa.csv`.
- Desempeño por daño: `resultados/metricas/moderador_grueso/desempeno_danos_modelo_ganador.csv`.
- Puerta de suficiencia: `resultados/metricas/moderador_grueso/puerta_suficiencia.csv`.
- Figura comparativa: `resultados/figuras/moderador_grueso/comparacion_modelos_y_f1_categoria.png`.
- Curva de revisión: `resultados/figuras/moderador_grueso/curva_revision_flags.png`.
- Modelo exportado: `modelos/moderador_grueso/moderador_cinco_danos_o_seguro.joblib`.
