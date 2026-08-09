# Análisis de calibración del `score_confianza` declarado por los LLM

**Fecha de corte:** 8 de agosto de 2026  
**Estado:** análisis concluido; recomendación no aplicada  
**Alcance:** etiquetado en cascada Flash–Pro, sin reentrenar los modelos y sin modificar cuadernos, umbrales ni datos

## Resumen ejecutivo

El campo `score_confianza` del etiquetado es una **autoevaluación numérica emitida por el propio LLM** dentro del JSON. El contrato exige que pertenezca al intervalo [0, 1], pero no lo convierte por sí solo en una probabilidad estadísticamente calibrada. La literatura muestra que las confianzas verbalizadas por un LLM pueden evaluarse y recalibrarse, aunque su validez depende de la tarea, el modelo y el protocolo de elicitación [1].

En el panel local de calibración hay **1,000 pares Flash–Pro**, correspondientes a **1,000 `chunk_id` y 1,000 `video_id` distintos**. El evento analizado fue la coincidencia exacta entre los conjuntos de etiquetas gruesas de Flash y Pro. La concordancia exacta observada fue 0.67; la concordancia binaria daño/seguro fue 0.92. En el subconjunto operativo de **640 casos** que Flash marcó como seguros y sin `needs_review`, estas proporciones fueron 0.73 y 0.99, respectivamente.

Los resultados responden directamente a la pregunta planteada:

- Elevar la confianza al cuadrado, `s²`, **no mejora el MAE**: lo aumenta de 0.37 a 0.39. El aumento pareado estimado es 0.023, con IC bootstrap del 95 % [0.015, 0.030].
- `s²` sí mejora métricas propias para probabilidades: el Brier baja de 0.26 a 0.24 y la pérdida logarítmica de 0.81 a 0.70. Esto indica que el puntaje original es, en promedio, demasiado alto para representar la probabilidad de coincidencia exacta con Pro.
- La raíz cuadrada reduce el MAE a 0.35, pero empeora Brier, pérdida logarítmica y error de calibración. Esa aparente mejora del MAE es engañosa y se explica por el desbalance del evento; el MAE no es una regla de puntuación estrictamente propia para probabilidades [2], [3].
- En validación cruzada, la calibración beta obtuvo el mejor Brier global, 0.20, pero la relación aprendida fue no monótona y poco confiable en niveles con pocos ejemplos. No se recomienda incorporarla todavía al enrutamiento.
- En los 640 casos seguros elegibles, una transformación monótona próxima a `s^4.6` produjo un Brier de 0.19 frente a 0.23 sin calibrar. Este resultado solo estima **acuerdo con Pro**, no corrección frente a jueces humanos.

La recomendación es conservar `score_confianza` como dato original y, si se aprueba una fase posterior, agregar un campo distinto —por ejemplo, `probabilidad_acuerdo_pro_calibrada`—. Antes de usarlo para decisiones operativas debe validarse contra un conjunto humano independiente y representativo.

## 1. Pregunta de análisis y estimando

Sea `s_i` el `score_confianza` declarado por Flash para el caso `i`. Se definió como objetivo:

\[
y_i = \mathbf{1}\{L_i^{Flash}=L_i^{Pro}\},
\]

donde `L` es el conjunto normalizado de etiquetas gruesas. Por tanto, el estimando es:

\[
P(L^{Flash}=L^{Pro}\mid s),
\]

no `P(Flash es correcto | s)`.

Esta diferencia es central: Pro actúa como referencia automática de segunda etapa, pero no como verdad de terreno humana. El análisis permite estudiar si `s` anticipa el acuerdo Flash–Pro; no demuestra exactitud semántica, validez cultural ni seguridad de moderación.

Como contraste secundario se calculó la coincidencia binaria daño/seguro. No se utilizó como objetivo principal porque la salida requerida conserva categorías de daño más específicas.

## 2. Naturaleza de la “confianza” del LLM

En este proyecto, `score_confianza` tiene tres propiedades verificables:

1. Es generado por el LLM como parte de la respuesta estructurada.
2. El esquema únicamente exige un valor numérico entre 0 y 1.
3. El prompt aplica reglas operativas: evidencia insuficiente, `contexto_necesario`, `ironia_ambigua` o ciertos indicadores de revisión limitan o condicionan el puntaje.

Por ello, el valor combina juicio del modelo y restricciones del prompt. No es un logit, una probabilidad de token ni una probabilidad calibrada mediante frecuencias observadas. Trabajos sobre confianza verbalizada muestran que pedir al modelo una probabilidad puede producir señales útiles, pero la calibración debe comprobarse empíricamente para cada distribución de uso [1].

Una transformación posterior puede cambiar la interpretación numérica y corregir desajustes sistemáticos, pero no añade información ni mejora el orden de los casos si es estrictamente monótona. En particular, `s²`, `sqrt(s)` y `s^a` preservan el ranking y, por tanto, el AUC.

## 3. Fuentes de datos y trazabilidad

Los resultados propios proceden exclusivamente de estos artefactos versionados del proyecto:

- [`calibration_flash.jsonl`](../datos/etiquetado/cascada_deepseek_v4/calibration_flash.jsonl): salidas de Flash.
- [`calibration_pro.jsonl`](../datos/etiquetado/cascada_deepseek_v4/calibration_pro.jsonl): salidas de Pro para los mismos casos.
- [`calibration_flash_vs_pro.json`](../datos/etiquetado/cascada_deepseek_v4/calibration_flash_vs_pro.json): resumen canónico de la comparación.
- [`prompt_operacional_ollama_v2.md`](../config/prompt_operacional_ollama_v2.md): reglas que condicionan la respuesta y el puntaje.
- [`schemas.py`](../src/moderacion_peru/schemas.py): contrato estructurado y límites del campo.
- [`labeling_calibration.py`](../src/moderacion_peru/labeling_calibration.py): lógica local de comparación Flash–Pro.

Los archivos internos documentan el procedimiento y los resultados propios; no se presentan como fuentes bibliográficas externas.

## 4. Método

### 4.1 Emparejamiento y subconjuntos

Se emparejaron las salidas por `chunk_id`. El panel contiene 1,000 pares completos, sin duplicados de `chunk_id` ni de `video_id`. Se evaluaron:

- **Panel completo:** 1,000 casos.
- **Seguros elegibles:** 640 casos que Flash marcó como seguros y sin `needs_review`, conforme a la regla operativa de la cascada.

### 4.2 Transformaciones evaluadas

Se comparó el puntaje original con transformaciones deterministas:

\[
g(s)\in\{s,\ s^2,\ s^3,\ \sqrt{s},\ s^a\}.
\]

También se evaluaron, fuera de muestra, calibración logística o de Platt [4], regresión isotónica [5] y calibración beta [6]. Para `s^a`, el exponente se seleccionó dentro de cada partición de entrenamiento, minimizando Brier o pérdida logarítmica según la variante.

### 4.3 Validación

La evaluación principal utilizó validación cruzada de cinco particiones. La calibración se ajustó únicamente con las particiones de entrenamiento y se predijo sobre la partición excluida. La separación se agrupó por `video_id`, aunque en este panel cada video aporta un solo caso.

La incertidumbre de las diferencias pareadas se estimó mediante 10,000 remuestras bootstrap [8]. Esta remuestra conserva en cada iteración la comparación de todos los métodos sobre los mismos casos.

### 4.4 Métricas

Para probabilidades `p_i=g(s_i)` y resultados binarios `y_i`, se calcularon:

\[
MAE=\frac{1}{n}\sum_i|p_i-y_i|,
\]

\[
Brier=\frac{1}{n}\sum_i(p_i-y_i)^2,
\]

\[
LogLoss=-\frac{1}{n}\sum_i\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right].
\]

El Brier y la pérdida logarítmica son reglas propias: en esperanza, incentivan reportar la probabilidad verdadera [2], [3]. El MAE no posee esa propiedad para una variable binaria. Si la prevalencia de `y=1` supera 0.5, el MAE favorece empujar las predicciones hacia 1, incluso aunque la probabilidad resultante esté peor calibrada.

Se añadió el error esperado de calibración con diez intervalos:

\[
ECE_{10}=\sum_{b=1}^{10}\frac{|B_b|}{n}\,|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)|.
\]

El ECE se usa como diagnóstico complementario; depende de la partición en intervalos y no sustituye las reglas propias. La práctica moderna de evaluación de calibración suele combinar diagramas o intervalos de confiabilidad, ECE y métodos posteriores de calibración [7]. En todas estas métricas, valores menores son mejores, salvo AUC, donde valores mayores indican mejor discriminación.

## 5. Resultados

### 5.1 Distribución observada

**Tabla 1. Resultado propio: frecuencia y concordancia exacta Flash–Pro en niveles con mayor soporte o interés operativo.**

| `score_confianza` | Casos | Acuerdo exacto, panel completo | Casos seguros elegibles | Acuerdo exacto, seguros elegibles |
|---:|---:|---:|---:|---:|
| 0.65 | 145 | 0.54 | 0 | — |
| 0.80 | 30 | 0.33 | 2 | 0.50 |
| 0.85 | 71 | 0.46 | 28 | 0.54 |
| 0.90 | 241 | 0.56 | 176 | 0.57 |
| 0.95 | 454 | 0.79 | 424 | 0.80 |
| 0.98 | 7 | 1.00 | 6 | 1.00 |

El nivel 0.95 concentra 454 de los 1,000 casos. Los niveles extremos tienen poco soporte; por ejemplo, solo siete casos presentan 0.98. En consecuencia, no debe interpretarse una tasa de 1.00 en esos niveles como una estimación precisa.

La secuencia tampoco es monótona en todos los puntos: la concordancia de 0.65 supera la observada en 0.80 y 0.85. Parte de esta forma se relaciona con las reglas del prompt y con composiciones de casos distintas, no necesariamente con una escala probabilística continua.

### 5.2 Transformaciones fijas en el panel completo

**Tabla 2. Resultado propio: evaluación descriptiva sobre los 1,000 pares.**

| Puntaje | MAE | Brier | Pérdida logarítmica | `ECE_10` | AUC |
|---|---:|---:|---:|---:|---:|
| `s` | 0.37 | 0.26 | 0.81 | 0.22 | 0.63 |
| `s²` | 0.39 | 0.24 | 0.70 | 0.18 | 0.63 |
| `s³` | 0.41 | 0.24 | 0.69 | 0.15 | 0.63 |
| `sqrt(s)` | 0.35 | 0.29 | 0.97 | 0.27 | 0.63 |

`s²` reduce Brier, pérdida logarítmica y ECE, pero empeora el MAE. La raíz cuadrada muestra el patrón opuesto. Como todas son transformaciones monótonas, el AUC permanece en 0.63: cambia la escala, no la capacidad de ordenar casos.

### 5.3 Calibración fuera de muestra: panel completo

**Tabla 3. Resultado propio: predicciones fuera de muestra mediante cinco particiones.**

| Método | MAE | Brier | Pérdida logarítmica | `ECE_10` |
|---|---:|---:|---:|---:|
| Sin calibrar | 0.37 | 0.26 | 0.81 | 0.22 |
| `s²` | 0.39 | 0.24 | 0.70 | 0.18 |
| `s³` | 0.41 | 0.24 | 0.69 | 0.15 |
| Potencia ajustada por Brier | 0.40 | 0.24 | 0.69 | 0.16 |
| Potencia ajustada por pérdida logarítmica | 0.40 | 0.24 | 0.69 | 0.16 |
| Beta | 0.41 | 0.20 | 0.59 | 0.013 |
| Isotónica | 0.42 | 0.21 | 0.60 | 0.00097 |

La calibración beta logra la mayor reducción del Brier global. Sin embargo, el ajuste completo aprendido es no monótono en la zona baja y media: refleja que los niveles de confianza están condicionados por reglas y tipos de caso diferentes. La regresión isotónica produce un ECE muy bajo, pero aplana gran parte del rango; un ECE bajo no implica mayor discriminación ni valida por sí solo el puntaje.

### 5.4 Subconjunto operativo de seguros elegibles

**Tabla 4. Resultado propio: cinco particiones sobre 640 casos seguros y sin revisión solicitada por Flash.**

| Método | MAE | Brier | Pérdida logarítmica | `ECE_10` |
|---|---:|---:|---:|---:|
| Sin calibrar | 0.30 | 0.23 | 0.76 | 0.20 |
| `s²` | 0.32 | 0.21 | 0.63 | 0.14 |
| `s³` | 0.34 | 0.19 | 0.58 | 0.085 |
| Potencia ajustada por Brier | 0.38 | 0.19 | 0.56 | 0.024 |
| Potencia ajustada por pérdida logarítmica | 0.37 | 0.19 | 0.56 | 0.023 |
| Beta | 0.37 | 0.19 | 0.56 | 0.018 |
| Isotónica | 0.37 | 0.19 | 0.57 | 0.016 |

Los exponentes seleccionados por Brier oscilaron entre 4.6 y 4.8; los seleccionados por pérdida logarítmica, entre 4.5 y 4.7. La estabilidad entre particiones permite resumir la transformación preliminar como `s^4.6`. Por ejemplo:

\[
0.95^{4.6}\approx 0.79, \qquad 0.85^{4.6}\approx 0.47.
\]

El primer valor coincide de cerca con la tasa empírica de acuerdo exacto, 0.80, observada entre los seguros elegibles con confianza 0.95. Esta correspondencia es descriptiva y no constituye una validación humana.

### 5.5 Incertidumbre de comparaciones clave

**Tabla 5. Resultado propio: cambio respecto del puntaje sin calibrar; valores negativos son mejoras para cada error.**

| Comparación | Métrica | Cambio | IC bootstrap del 95 % | Lectura |
|---|---|---:|---:|---|
| `s²` frente a `s`, panel completo | MAE | +0.023 | [+0.015, +0.030] | Empeora |
| `s²` frente a `s`, panel completo | Brier | −0.019 | [−0.027, −0.011] | Mejora |
| `sqrt(s)` frente a `s`, panel completo | MAE | −0.014 | [−0.019, −0.0092] | Mejora solo MAE |
| Potencia ajustada frente a `s`, seguros elegibles | Brier | −0.048 | [−0.064, −0.032] | Mejora |

Los intervalos cuantifican variabilidad por remuestreo dentro de este panel. No cubren cambios de distribución, errores compartidos entre Flash y Pro ni desacuerdo con evaluadores humanos.

## 6. Interpretación

### 6.1 ¿Conviene elevar la confianza al cuadrado?

Depende de la afirmación que se quiera sostener:

- **Para mejorar MAE:** no. `s²` aumenta el MAE de manera consistente en este panel.
- **Para aproximar una probabilidad de acuerdo exacto con Pro:** sí hay evidencia favorable, porque mejora Brier, pérdida logarítmica y ECE.
- **Para demostrar corrección real del etiquetado:** no. El objetivo observado es Pro, no una adjudicación humana independiente.

### 6.2 ¿Por qué la raíz cuadrada “mejora” MAE pero no es aconsejable?

La raíz eleva todos los valores de `s` situados entre 0 y 1. Como el acuerdo exacto ocurre en 0.67 del panel, el MAE premia acercar las predicciones a la clase mayoritaria `y=1`. Esta reducción no significa que las cifras sean probabilidades más fieles: Brier pasa de 0.26 a 0.29 y la pérdida logarítmica de 0.81 a 0.97.

### 6.3 ¿Qué aporta `s^4.6`?

En el subconjunto que realmente podría aceptarse sin revisión, `s^4.6` es una aproximación monótona, sencilla y estable entre particiones a la probabilidad de acuerdo con Pro. Tiene tres ventajas frente a una calibración beta en esta etapa:

- mantiene el orden original;
- es fácil de auditar y reproducir;
- evita una curva no monótona difícil de justificar operativamente.

No obstante, el exponente está estimado sobre el panel disponible y puede cambiar con nuevos lotes, cambios de prompt, modelos, categorías o composición temática.

## 7. Efecto sobre los umbrales

Recalibrar sin transformar también el umbral modifica el volumen de revisión. Para conservar un umbral original `t` bajo una transformación `g`, debe emplearse `g(t)`:

- Con `g(s)=s²`, el umbral bruto 0.85 equivale a un umbral transformado de **0.72**.
- Con `g(s)=s^4.6`, el umbral bruto 0.85 equivale a un umbral transformado de **0.47**.

Si se aplicara `s²` y se conservara numéricamente el umbral 0.85, el umbral efectivo sobre la confianza original subiría a `sqrt(0.85)`, aproximadamente **0.92**. Con `s^4.6`, subiría aproximadamente a **0.97**. Esto enviaría más casos a revisión aunque el orden no cambie.

Por tanto, una calibración no debe introducirse sustituyendo silenciosamente el campo existente. El puntaje calibrado y su umbral necesitan nombres, trazabilidad y pruebas separados.

## 8. Validez y limitaciones

### 8.1 Validez de constructo

El evento mide acuerdo exacto con Pro. Flash y Pro pueden compartir sesgos, instrucciones, taxonomía o errores. Un acuerdo alto no equivale a corrección, y un desacuerdo no identifica automáticamente cuál salida es mejor.

### 8.2 Validez interna

La evaluación fuera de muestra reduce el optimismo de ajustar y medir sobre los mismos casos. Sin embargo, el análisis es observacional y depende de una sola campaña y versión operativa.

### 8.3 Validez externa

No se ha demostrado que la curva se mantenga ante nuevos canales, temas, períodos, prompts, proveedores o modelos. Los niveles con pocos casos tienen incertidumbre elevada.

### 8.4 Métricas

MAE puede inducir una selección inadecuada para probabilidades binarias. ECE depende del número y los límites de sus intervalos. Brier y pérdida logarítmica son más adecuados como criterios principales, pero ninguna métrica resuelve la ausencia de verdad humana.

### 8.5 Acciones no realizadas

Este análisis:

- no llamó a las API de DeepSeek;
- no reentrenó ni volvió a consultar Flash o Pro;
- no modificó etiquetas históricas;
- no cambió el cuaderno `02_01`, el prompt, los umbrales ni el enrutamiento;
- no afirma que el puntaje calibrado sea probabilidad de corrección humana.

## 9. Recomendación para decisión posterior

**Decisión local propuesta, todavía no aplicada:**

1. Mantener `score_confianza` sin alteraciones como salida original auditable.
2. Si se autoriza una prueba, crear un campo separado, como `probabilidad_acuerdo_pro_calibrada`.
3. Usar Brier y pérdida logarítmica como criterios principales; conservar MAE solo como diagnóstico secundario.
4. Evaluar primero una transformación monótona específica para los seguros elegibles, con `s^4.6` como candidato preliminar.
5. No adoptar aún la calibración beta para enrutamiento, pese a su mejor Brier global, debido a la no monotonicidad observada y al escaso soporte en varios niveles.
6. Congelar un conjunto humano de validación, con adjudicación y representación de categorías críticas, antes de interpretar el resultado como probabilidad de corrección.
7. Reestimar la calibración cuando cambien el modelo, prompt, taxonomía o distribución de datos.
8. Simular y documentar el efecto presupuestario de cualquier umbral antes de activarlo.

## 10. Conclusión

La confianza declarada por los LLM contiene señal, pero no está calibrada como probabilidad de acuerdo exacto con Pro. Elevarla al cuadrado mejora Brier y pérdida logarítmica, pero **no mejora MAE**. La aparente ventaja de la raíz cuadrada en MAE es incompatible con las métricas propias y no constituye evidencia de mejor calibración.

Para el subconjunto operativo de seguros elegibles, una potencia cercana a 4.6 ofrece una alternativa simple y auditable, con una mejora de Brier fuera de muestra de aproximadamente 0.048. Este hallazgo justifica una prueba controlada, no un cambio automático: todavía falta validar contra etiquetas humanas independientes y estimar el impacto de los nuevos umbrales sobre revisión, costo y riesgo.

## Referencias

[1] K. Tian, E. Mitchell, A. Zhou *et al*., “Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback,” in *Proc. 2023 Conf. Empirical Methods in Natural Language Processing*, 2023, pp. 5433–5442, doi: [10.18653/v1/2023.emnlp-main.330](https://doi.org/10.18653/v1/2023.emnlp-main.330).

[2] G. W. Brier, “Verification of Forecasts Expressed in Terms of Probability,” *Monthly Weather Review*, vol. 78, no. 1, pp. 1–3, 1950, doi: [10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2).

[3] T. Gneiting and A. E. Raftery, “Strictly Proper Scoring Rules, Prediction, and Estimation,” *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359–378, 2007, doi: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).

[4] J. C. Platt, “Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods,” in *Advances in Large Margin Classifiers*, A. J. Smola, P. L. Bartlett, B. Schölkopf, and D. Schuurmans, Eds. Cambridge, MA, USA: MIT Press, 1999, pp. 61–74.

[5] B. Zadrozny and C. Elkan, “Transforming Classifier Scores into Accurate Multiclass Probability Estimates,” in *Proc. 8th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2002, pp. 694–699, doi: [10.1145/775047.775151](https://doi.org/10.1145/775047.775151).

[6] M. Kull, T. Silva Filho, and P. Flach, “Beta Calibration: A Well-Founded and Easily Implemented Improvement on Logistic Calibration for Binary Classifiers,” in *Proc. 20th Int. Conf. Artificial Intelligence and Statistics*, vol. 54, 2017, pp. 623–631. [Online]. Available: [PMLR](https://proceedings.mlr.press/v54/kull17a.html).

[7] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in *Proc. 34th Int. Conf. Machine Learning*, vol. 70, 2017, pp. 1321–1330. [Online]. Available: [PMLR](https://proceedings.mlr.press/v70/guo17a.html).

[8] B. Efron, “Bootstrap Methods: Another Look at the Jackknife,” *The Annals of Statistics*, vol. 7, no. 1, pp. 1–26, 1979, doi: [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).

## Auditoría de citado

- Fuentes externas en la bibliografía: 8.
- Fuentes externas citadas en el texto: 8.
- Referencias bibliográficas sin uso: 0.
- Citas sin entrada bibliográfica: 0.
- Referencias duplicadas: 0.
- Resultados propios identificados como tales: sí.
- Decisiones locales identificadas como propuestas no aplicadas: sí.
- Pendientes bibliográficos: 0.
