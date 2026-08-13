# Criterio de selección de modelos de moderación

**Alcance:** candidatos individuales y *ensembles* de `03_01`–`03_06b`,
comparados y congelados por `03_07`.

**Estado:** criterio normativo para la comparación definitiva en `validation`.
No autoriza a abrir `test` ni a publicar un modelo.

Los candidatos con entrenamiento deliberadamente acotado pueden aparecer en
la tabla principal si producen predicciones sobre la `validation` común
completa, conservan el mismo snapshot y mantienen `test` sellado. Deben exponer
`training_regime`, presupuesto y disclaimer en el manifiesto y en el ranking.
La comparación describe el sistema efectivamente entrenado; no convierte una
corrida corta en evidencia del desempeño que tendría un ajuste exhaustivo.

## 1. Decisión recomendada

El ganador académico principal será el sistema que obtenga la mayor
**exactitud balanceada binaria** (*balanced accuracy*) al reducir las cinco
salidas a la decisión operativa `DAÑO` frente a `SEGURO`. Esta métrica da el
mismo peso a detectar daño y a reconocer contenido seguro, por lo que responde
directamente al objetivo local de equilibrar falsos seguros y falsos daños sin
quedar dominada por la prevalencia mayoritaria de `SEGURO` [1], [2].

La selección no debe basarse únicamente en *accuracy*, micro-F1, macro-F1 o
AUPRC. En problemas multietiqueta no existe una sola métrica que describa todas
las propiedades de una predicción [3]. Además, las curvas precisión--*recall* y
la AUPRC son especialmente informativas bajo desbalance [4], [5], pero miden la
calidad del ordenamiento de los *scores* a través de umbrales, no el costo de la
decisión operativa finalmente adoptada. Por ello:

1. la **balanced accuracy binaria** será el criterio primario de ranking;
2. la macro-AUPRC de los cuatro daños será una salvaguarda multietiqueta y una
   métrica secundaria obligatoria;
3. el sistema selectivo se evaluará mediante su curva riesgo--cobertura y carga
   de revisión, sin tratar `NEEDS_REVIEW` como una sexta etiqueta;
4. la superioridad se decidirá con incertidumbre pareada y agrupada por video,
   no solo con diferencias puntuales.

### 1.1. Por qué no se suman todas las métricas

No se construirá una puntuación aditiva con BA, FNR, FPR, F1, AUPRC,
calibración y carga de revisión. BA ya es una función de FNR y FPR; incluir las
tres cantidades en una suma contaría dos veces los mismos errores. Además,
AUPRC, ECE y carga humana representan propiedades distintas y sus escalas no
son utilidades intercambiables. Una suma ponderada solo sería defendible si sus
pesos provinieran de costos o preferencias explícitas de las partes interesadas
y se acompañara de análisis de sensibilidad.

El enfoque vigente es **lexicográfico con salvaguardas**: maximizar BA binaria
a cobertura completa; aplicar la no inferioridad macro-AUPRC predeclarada;
informar la frontera BA--macro-AUPRC; y usar menor (R_{0.67}), luego mayor
macro-AUPRC, únicamente como desempates. Esto es coherente con la literatura de
aprendizaje multiobjetivo, que distingue la agregación escalar de la selección
Pareto y exige explicitar preferencias para escoger entre compromisos [14].

## 2. Proyección binaria inequívoca

Sea \(Y_{i\ell}\) la referencia de la etiqueta de daño \(\ell\) para el chunk
\(i\), donde \(\ell\) recorre las cuatro categorías de daño. La verdad binaria
es:

\[
Y_i^{D}=\max_{\ell\in\mathcal D}Y_{i\ell}.
\]

Los *scores* de las cuatro categorías se calibran con el mismo protocolo para
todos los candidatos. Sea \(\widetilde s_{i\ell}\) el *score* calibrado. La
compuerta binaria usa:

\[
q_i^D=\max_{\ell\in\mathcal D}\widetilde s_{i\ell},
\qquad
\widehat Y_i^{D}=\mathbb 1[q_i^D\ge \tau_D].
\]

El único umbral \(\tau_D\) alinea la decisión binaria con BA o con el riesgo
coste-sensible predeclarado. Los umbrales \(t_\ell\) por categoría se conservan
separadamente para decidir **qué** daños informar. No se deben optimizar cuatro
umbrales para F1 por etiqueta y presentar después la BA resultante como si los
umbrales hubieran sido elegidos para minimizar el error balanceado.

La calibración y \(\tau_D\) se estimarán mediante *cross-fitting* agrupado por
`video_id` dentro de `validation`: cada fold se evalúa con calibradores y umbral
aprendidos sin sus videos. Las predicciones *out-of-fold* concatenadas producen
el ranking. Una vez elegido el sistema, el calibrador y los umbrales de
despliegue se reajustan sobre toda `validation` y se congelan antes de `test`.
Esta separación reduce el optimismo de escoger y evaluar umbrales en las mismas
filas [12].

La salida `SEGURO` no puede anular un daño detectado. En operación se aplicará
la precedencia siguiente:

1. si \(q_i^D\ge\tau_D\), la decisión binaria para el ranking es `DAÑO`;
2. si \(q_i^D<\tau_D\), la decisión binaria para el ranking es `SEGURO`;
3. si la compuerta binaria, la cabeza `SEGURO` y las categorías activas no son
   coherentes entre sí, no se emite una decisión automática y el caso se envía
   a `NEEDS_REVIEW`;
4. si la compuerta decide daño pero ninguna categoría supera su umbral, el caso
   también se revisa: el sistema detectó riesgo, pero no puede justificar qué
   categoría asignar.

Esta regla preserva el contrato: `SEGURO` es excluyente y los cuatro daños son
multietiqueta. Un conflicto `SEGURO + daño` es un fallo de coherencia o una señal
de incertidumbre, no una predicción válida de `SEGURO`.

## 3. Errores que entran al ranking

El **falso seguro binario** es omitir todos los daños en un chunk realmente
dañino:

\[
\operatorname{FNR}_{D}=
\frac{FN_D}{TP_D+FN_D}=1-\operatorname{Recall}_{D}.
\]

El **falso daño binario** es activar al menos un daño en un chunk realmente
seguro:

\[
\operatorname{FPR}_{D}=
\frac{FP_D}{TN_D+FP_D}.
\]

Con igual peso normativo para las dos clases, el error balanceado y la
exactitud balanceada son:

\[
\operatorname{BER}_{0.5}
=\tfrac{1}{2}(\operatorname{FNR}_{D}+\operatorname{FPR}_{D}),
\qquad
\operatorname{BA}=1-\operatorname{BER}_{0.5}
=\tfrac{1}{2}(\operatorname{TPR}_{D}+\operatorname{TNR}_{D}).
\]

Brodersen *et al.* proponen la balanced accuracy como el promedio de la
exactitud obtenida en cada clase y muestran por qué es apropiada cuando una
accuracy ordinaria puede favorecer al grupo mayoritario [1]. Este es el
significado de “respuesta balanceada” adoptado por el proyecto.

### 3.1. Métrica que no debe confundirse con el falso seguro binario

El artefacto vigente también guarda `false_safe_rate_on_damage`, que cuenta
casos dañinos donde la cabeza `SEGURO` quedó activa. Esa cifra incluye conflictos
en los que el modelo **también detectó un daño**. Sirve para auditar la
exclusividad, pero no es la FNR binaria y no se sumará a
`false_alarm_rate_on_safe` para elegir al ganador.

Los artefactos actuales permiten una lectura provisional mediante:

- `FNR_D = 1 - validation_metrics.any_damage.recall`;
- `FPR_D = validation_metrics.false_alarm_rate_on_safe`;
- `BA = validation_metrics.any_damage.balanced_accuracy`.

La comparación definitiva deberá recalcular esos tres campos con la compuerta
\(q_i^D,\tau_D\) y el *cross-fitting* descritos arriba. Hasta entonces no debe
suponerse que los umbrales vigentes, ajustados por F1 de cada etiqueta,
implementan ya el nuevo criterio.

## 4. `NEEDS_REVIEW` como política selectiva

`NEEDS_REVIEW` **no es una salida aprendida ni una sexta clase**. Es una acción
determinista posterior a la inferencia que deriva de los cinco *scores*, los
umbrales congelados y una zona de incertidumbre. Esta separación coincide con
la clasificación selectiva o con opción de rechazo: el sistema puede abstenerse
de decidir automáticamente a cambio de menor riesgo y menor cobertura [6]. La
derivación a una persona también es consistente con el marco más general de
*learning to defer*, aunque aquí no se entrena todavía un predictor específico
del desempeño del revisor [7].

Un chunk tendrá `needs_review=true` si ocurre al menos una condición:

1. **conflicto:** se activan `SEGURO` y uno o más daños;
2. **salida vacía:** ninguna de las cinco salidas supera su umbral;
3. **incertidumbre de umbral:** \(q_i^D\) cae dentro de una banda \(\delta_D\)
   alrededor de \(\tau_D\), o una categoría relevante cae dentro de su banda
   \(\delta_\ell\) alrededor de \(t_\ell\);
4. **regla de seguridad aprobada:** una señal adicional predeclarada, como una
   combinación de baja confianza y categoría de alto impacto.

Las bandas \(\delta_D,\delta_\ell\) no vienen fijadas por la literatura y son
decisiones locales.
El valor actual `0.05` es un punto de partida implementado, no una constante
universal. Su valor definitivo debe elegirse una sola vez en `validation`,
después de calibrar los *scores* y antes de abrir `test`. Las redes modernas
pueden estar descalibradas; por ello se deben conservar ECE, Brier y diagramas
de confiabilidad junto con cualquier umbral que pretenda representar
incertidumbre [8].

Para una política \(\boldsymbol\delta\), se reportarán:

\[
\operatorname{Cobertura}(\boldsymbol\delta)=
\frac{N_{\text{decisiones automáticas}}}{N_{\text{total}}},
\]

\[
\operatorname{BER}_{\text{selectivo}}(\boldsymbol\delta)=
\tfrac{1}{2}
\left(\operatorname{FNR}_{D\mid\text{auto}}+
\operatorname{FPR}_{D\mid\text{auto}}\right).
\]

Debe mostrarse la curva riesgo--cobertura completa o una cuadrícula
predeclarada de \(\boldsymbol\delta\). No es válido comparar el error selectivo sin
informar cobertura: un modelo podría aparentar ser perfecto enviando casi todo
a revisión. Tampoco se asumirá que el revisor humano es infalible; cuando haya
datos pareados suficientes, se reportará por separado el desempeño del sistema
modelo--humano [7].

### 4.1. Uso en la selección

La balanced accuracy **a cobertura completa**, calculada antes de retirar los
casos enviados a revisión, seguirá siendo el ranking primario. Así se impide que
la abstención mejore artificialmente la posición de un candidato.

La política `NEEDS_REVIEW` se elegirá después, entre los puntos de la curva
riesgo--cobertura del modelo seleccionado, mediante una restricción operativa
aprobada de cobertura o capacidad humana. Mientras no exista una capacidad
formal de revisión, se informará la frontera y no se inventará un porcentaje
mínimo respaldándolo falsamente en la literatura.

## 5. Costos asimétricos y análisis de sensibilidad

La balanced accuracy supone igual importancia para ambas clases. No supone que
ambos errores tengan igual costo económico o social en despliegue. La teoría de
clasificación sensible a costos recomienda decidir con una matriz de costos
coherente cuando una confusión es más grave que la otra [9]. Como el proyecto
todavía no dispone de costos validados con usuarios, moderadores y capacidad de
revisión, el ranking principal conservará pesos iguales y añadirá el análisis:

\[
R_\lambda =
\lambda\operatorname{FNR}_{D}+
(1-\lambda)\operatorname{FPR}_{D},
\qquad 0\le\lambda\le1.
\]

Se reportarán como mínimo \(\lambda\in\{0.50, 0.67, 0.80\}\): igualdad; peso
normalizado del falso seguro aproximadamente doble; y aproximadamente
cuádruple. Estos dos últimos escenarios son análisis de sensibilidad locales,
**no costos derivados de las referencias**. Si cambia el ganador según
\(\lambda\), el informe debe declararlo y mostrar la frontera FNR--FPR en lugar
de afirmar una superioridad universal.

Cuando existan prevalencias de despliegue y costos sustentados, el análisis
operativo deberá usar el costo esperado, no \(R_\lambda\) aislado:

\[
\mathbb E[C]=
\pi_D C_{FN}\operatorname{FNR}_D+
(1-\pi_D)C_{FP}\operatorname{FPR}_D+
C_R\Pr(\text{NEEDS_REVIEW}),
\]

donde \(\pi_D\) es la prevalencia esperada de daño y \(C_R\) incluye el costo de
revisión. Si se conoce el error humano residual, también debe incorporarse al
costo del sistema completo. La prevalencia 4:1 de `validation` es un benchmark
de selección y no debe presentarse automáticamente como prevalencia de
producción.

## 6. Salvaguardas multietiqueta obligatorias

La proyección binaria no basta para seleccionar un moderador útil. Un sistema
podría detectar “algún daño” y fallar siempre en una categoría minoritaria. Todo
candidato deberá acompañar la BA con:

- macro-AUPRC de los cuatro daños y AUPRC por daño;
- macro-F1 y *recall* por daño al umbral congelado;
- F1 y AUPRC de `any_damage`;
- tasa de conflicto `SEGURO + daño`;
- ECE y Brier por salida;
- macro-F1 y macro-AUPRC de etiquetas finas solo en posiciones observadas;
- carga de revisión y curva riesgo--cobertura;
- métricas por canal o en la partición de canales retenidos cuando corresponda.

La macro-AUPRC de daño actuará como **salvaguarda de no degradación**, no como
un segundo objetivo ajustable después de ver resultados. Antes de ejecutar
`03_07` debe fijarse un margen de no inferioridad frente al mejor individuo
completo. Si el margen todavía no ha sido aprobado, se reportará la frontera
BA--macro-AUPRC y no se excluirán modelos mediante un margen retrospectivo.

## 7. Incertidumbre, empate y declaración de ganador

Los chunks de un mismo video no son observaciones independientes. Los intervalos
y diferencias se estimarán con *bootstrap* pareado agrupado por `video_id`,
conservando juntos todos los chunks del video [10]. La comparación de PLN debe
adecuar la prueba a la métrica y al diseño pareado [11]. Optimizar repetidamente
una estimación ruidosa de `validation` puede sobreajustar el propio criterio de
selección, de modo que el criterio, sus pesos y sus desempates deben quedar
congelados antes de abrir `test` [12].

Procedimiento:

1. admitir solo candidatos completos del mismo SHA-256, contrato y partición,
   con `test_metrics=null`;
2. generar decisiones *out-of-fold* con calibración y \(\tau_D\) aprendidos en
   folds agrupados por video dentro de `validation`;
3. calcular BA, FNR, FPR y métricas multietiqueta para cada individuo y
   *ensemble*;
4. ordenar por BA puntual decreciente;
5. comparar el primero con cada retador mediante diferencias de BA en al menos
   2.000 réplicas pareadas agrupadas por video y un IC del 95 %;
6. si se realizan múltiples contrastes, corregir los valores *p* mediante Holm
   o aplicar una regla de intervalos simultáneos predeclarada [13];
7. declarar **ganador confirmado en validation** solo si el IC del 95 % de
   \(\Delta BA=BA_{\text{líder}}-BA_{\text{retador}}\) queda por encima de cero
   frente al mejor retador elegible y se cumplen las salvaguardas;
8. si el intervalo incluye cero, declarar **empate estadístico/no concluyente**.
   Dentro del empate se prefiere, en este orden, menor \(R_{0.67}\), mayor
   macro-AUPRC de daño, mayor cobertura a igual riesgo y menor costo de
   inferencia. Esa preferencia operativa no debe describirse como superioridad
   estadística.

`Test` se abre una sola vez después de congelar modelo, *ensemble*, umbrales y
política `NEEDS_REVIEW`. Sus resultados estiman generalización y no permiten
volver a rankear, reajustar \(\boldsymbol\delta\), cambiar \(\lambda\) ni escoger
otro modelo.

## 8. Lectura provisional de los resultados disponibles

Con los candidatos completos actualmente visibles y la proyección binaria
anterior, las estimaciones puntuales de `validation` 4:1 son:

| Candidato | FNR daño | FPR sobre seguro | BER con peso 1:1 por clase | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Qwen LoRA | 0,3142 | 0,0880 | **0,2011** | **0,7989** |
| Multitarea | 0,3302 | 0,0929 | 0,2116 | 0,7884 |
| E5 plano | 0,3354 | 0,0915 | 0,2134 | 0,7866 |
| Cascada v2 | 0,3599 | **0,0804** | 0,2202 | 0,7798 |
| MiniLM plano | 0,3585 | 0,0863 | 0,2224 | 0,7776 |

Estas cifras reutilizan los umbrales por etiqueta del artefacto vigente; aún no
son estimaciones *out-of-fold* de la nueva compuerta binaria. Bajo esa lectura
provisional, Qwen LoRA ocupa el primer lugar puntual. No es todavía un ganador
confirmado: `03_06` no ha producido un candidato final y faltan la recalibración
alineada, el *cross-fitting* y el *bootstrap* pareado de BA de `03_07`. La tabla
tampoco sustituye el análisis multietiqueta ni la apertura única del test
natural.

## 9. Implicaciones para la implementación

La implementación vigente de `03_07` aplica este documento: cross-fitting
agrupado por video, BA binaria como ranking primario, frontera y salvaguarda
macro-AUPRC, desempate por (R_{0.67}), bootstrap pareado de BA y política
`NEEDS_REVIEW` posterior. Si la capacidad humana o el margen de no inferioridad
permanecen en `None`, genera el informe exploratorio pero mantiene `test`
sellado; no inventa esos valores.

El artefacto de comparación deberá persistir al menos:

- versión e identificador de este criterio;
- fórmula y pesos \(\lambda\) evaluados;
- BA, FNR y FPR a cobertura completa;
- definición, \(\boldsymbol\delta\), cobertura y riesgo de `NEEDS_REVIEW`;
- frontera BA--macro-AUPRC y riesgo--cobertura;
- IC pareados, agrupamiento, semilla, réplicas y corrección múltiple;
- salvaguardas, desempates y cualquier costo operativo usado;
- confirmación de que `test` permaneció sellado durante la selección.

## Referencias

[1] K. H. Brodersen, C. S. Ong, K. E. Stephan y J. M. Buhmann, “The
Balanced Accuracy and Its Posterior Distribution,” en *20th International
Conference on Pattern Recognition*, 2010, pp. 3121–3124. doi:
[10.1109/ICPR.2010.764](https://doi.org/10.1109/ICPR.2010.764).

[2] M. Sokolova y G. Lapalme, “A Systematic Analysis of Performance Measures
for Classification Tasks,” *Information Processing & Management*, vol. 45,
no. 4, pp. 427–437, 2009. doi:
[10.1016/j.ipm.2009.03.002](https://doi.org/10.1016/j.ipm.2009.03.002).

[3] M.-L. Zhang y Z.-H. Zhou, “A Review on Multi-Label Learning Algorithms,”
*IEEE Transactions on Knowledge and Data Engineering*, vol. 26, no. 8,
pp. 1819–1837, 2014. doi:
[10.1109/TKDE.2013.39](https://doi.org/10.1109/TKDE.2013.39).

[4] T. Saito y M. Rehmsmeier, “The Precision-Recall Plot Is More Informative
than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,”
*PLOS ONE*, vol. 10, no. 3, e0118432, 2015. doi:
[10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).

[5] J. Davis y M. Goadrich, “The Relationship Between Precision-Recall and ROC
Curves,” en *Proc. 23rd ICML*, 2006, pp. 233–240. doi:
[10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874).

[6] Y. Geifman y R. El-Yaniv, “Selective Classification for Deep Neural
Networks,” en *Advances in Neural Information Processing Systems*, vol. 30,
2017. [En línea](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).

[7] H. Mozannar y D. Sontag, “Consistent Estimators for Learning to Defer to
an Expert,” en *Proc. 37th ICML*, PMLR 119, pp. 7076–7087, 2020.
[En línea](https://proceedings.mlr.press/v119/mozannar20b.html).

[8] C. Guo, G. Pleiss, Y. Sun y K. Q. Weinberger, “On Calibration of Modern
Neural Networks,” en *Proc. 34th ICML*, PMLR 70, pp. 1321–1330, 2017.
[En línea](https://proceedings.mlr.press/v70/guo17a.html).

[9] C. Elkan, “The Foundations of Cost-Sensitive Learning,” en *Proc. 17th
IJCAI*, 2001, pp. 973–978.
[En línea](https://cseweb.ucsd.edu/~elkan/rescale.pdf).

[10] C. A. Field y A. H. Welsh, “Bootstrapping Clustered Data,” *Journal of
the Royal Statistical Society: Series B*, vol. 69, no. 3, pp. 369–390, 2007.
doi:
[10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x).

[11] R. Dror, G. Baumer, S. Shlomov y R. Reichart, “The Hitchhiker's Guide to
Testing Statistical Significance in Natural Language Processing,” en *Proc.
56th ACL*, 2018, pp. 1383–1392. doi:
[10.18653/v1/P18-1128](https://doi.org/10.18653/v1/P18-1128).

[12] G. C. Cawley y N. L. C. Talbot, “On Over-Fitting in Model Selection and
Subsequent Selection Bias in Performance Evaluation,” *Journal of Machine
Learning Research*, vol. 11, pp. 2079–2107, 2010.
[En línea](https://www.jmlr.org/papers/v11/cawley10a.html).

[13] S. Holm, “A Simple Sequentially Rejective Multiple Test Procedure,”
*Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979. doi:
[10.2307/4615733](https://doi.org/10.2307/4615733).

[14] Y. Jin y B. Sendhoff, “Pareto-Based Multiobjective Machine Learning: An
Overview and Case Studies,” *IEEE Transactions on Systems, Man, and
Cybernetics, Part C*, vol. 38, no. 3, pp. 397–415, 2008. doi:
[10.1109/TSMCC.2008.919172](https://doi.org/10.1109/TSMCC.2008.919172).
