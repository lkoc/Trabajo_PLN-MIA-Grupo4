# Criterios para seleccionar los modelos Transformer del moderador

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Fecha de decisión: 27 de julio de 2026  
Ámbito: clasificación multietiqueta de cinco categorías gruesas de daño; `SEGURO` se deriva cuando ninguna categoría de daño supera su umbral. Las etiquetas finas y las categorías transversales no son objetivos de entrenamiento.

## 1. Decisión

Se compararán dos encoders compactos:

1. **Modelo inicial:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
2. **Segundo modelo:** `intfloat/multilingual-e5-small`, revisión `614241f622f53c4eeff9890bdc4f31cfecc418b3`.

El segundo checkpoint pertenece funcionalmente al mismo estrato de cómputo: su ficha declara que fue inicializado desde `microsoft/Multilingual-MiniLM-L12-H384`. Por ello se lo denomina aquí **E5-small de linaje MiniLM**. No se afirma de antemano que será ganador; se formula una hipótesis contrastable de que su preentrenamiento contrastivo multilingüe más amplio puede mejorar la representación de los chunks.

Hasta la fecha de este documento se ejecutaron la descarga y los microbenchmarks del primer modelo, **no** el fine-tuning completo. Esta distinción evita reportar como resultado una ejecución que todavía no ha ocurrido.

## 2. Datos y restricción computacional

Las dos campañas canónicas reúnen 91.844 chunks crudos únicos. Tras excluir 1.870 pendientes o rechazados quedan 89.974 utilizables: 4.008 con algún daño y 85.966 `SEGURO`. Primero se conservan los 4.008 daños y se toman 16.032 seguros mediante hash reproducible; después se divide la muestra global de 20.040 chunks por video en train/validación/test 70/15/15. El resultado actual es 14.064/2.992/2.984 chunks, sin videos compartidos. En entrenamiento quedan 2.786 chunks con daño; la categoría menos representada es `AMENAZA_DIRECTA`, con 212 positivos.

El equipo tiene un AMD Ryzen 7 8845HS (8 núcleos, 16 hilos), 28,83 GB de RAM y PyTorch 2.6.0 sólo para CPU. Las Radeon RX 570 y 780M no aparecen en la matriz oficial vigente de PyTorch + ROCm para Windows. PyTorch tampoco incluye OpenCL entre sus backends oficiales. Por reproducibilidad, el experimento se ejecutará en CPU.

Microbenchmark observado con el primer modelo, longitud máxima de 128 tokens:

| Operación | Resultado observado |
|---|---:|
| Encoder congelado, lote 32 | 74,00 chunks/s |
| Codificación estimada de los 20.040 chunks balanceados | 4,5 min |
| Fine-tuning, lote 8 | 12,27 chunks/s |
| Una época del train de 14.064 chunks | 19,1 min |
| Tres épocas, sólo pasos de optimización | 57,3 min |
| Tres épocas con validación y checkpoints | 1–1,5 h por modelo, estimado |

## 3. Criterios de selección

### 3.1 Adecuación a la tarea

El problema es multietiqueta: un chunk puede activar varias de las cinco categorías de daño. Se necesita un encoder bidireccional seguido de cinco salidas sigmoidales y pérdida BCE. MiniLM conserva la arquitectura de autoatención de los Transformers mediante destilación, reduciendo el coste de ajuste y servicio (Wang et al., 2020). El fine-tuning adapta pesos preentrenados al dominio y requiere menos datos y cómputo que entrenar desde inicialización aleatoria (Hugging Face, 2026a).

GPT-2 no se prioriza. Aunque puede adaptarse a clasificación, es un modelo causal concebido para predecir el siguiente token. Para este problema resulta menos directo que un encoder bidireccional compacto y no ofrece una ventaja computacional o lingüística demostrada en el corpus peruano.

### 3.2 Español y español peruano

Ambos candidatos incluyen español dentro de su entrenamiento multilingüe. Sin embargo, ninguno fue preentrenado específicamente con español peruano. Por tanto, la procedencia lingüística constituye un criterio de plausibilidad, no evidencia suficiente: el desempeño debe decidirse con los chunks locales y la partición por video.

BETO permanece como tercer candidato futuro porque fue preentrenado exclusivamente en un corpus grande en español y reportó resultados competitivos frente a BERT multilingüe en varias tareas españolas (Cañete et al., 2020). No se elige ahora porque su dimensión oculta de 768 supone bastante más cómputo que los modelos compactos de 384 dimensiones y el entorno carece de GPU compatible.

### 3.3 Transferencia semántica

`paraphrase-multilingual-MiniLM-L12-v2` produce embeddings de 384 dimensiones y fue optimizado para similitud/paráfrasis en 50 lenguas. La formulación SBERT produce representaciones de oración útiles como características semánticas eficientes (Reimers & Gurevych, 2019). Es una primera opción conservadora, conocida y rápida.

`multilingual-e5-small` parte de Multilingual MiniLM, admite hasta 512 tokens y declara cobertura de 100 lenguas. Su entrenamiento incluye preentrenamiento contrastivo con aproximadamente mil millones de pares multilingües, seguido de ajuste supervisado; E5 fue concebido para transferir a recuperación, agrupamiento y clasificación (Wang et al., 2022, 2024). Por ese entrenamiento adicional es el segundo candidato con mayor expectativa previa, manteniendo aproximadamente el mismo tamaño de 0,1 mil millones de parámetros.

El contexto máximo es una ventaja potencial de E5, pero la comparación principal fijará **128 tokens para ambos modelos** a fin de aislar el efecto del checkpoint. Sólo después se hará, si se justifica, una ablación E5 a 256 tokens. Comparar inicialmente 128 frente a 256 confundiría calidad del modelo con cantidad de texto observada.

### 3.4 Desbalance y dependencia entre etiquetas

La clase `SEGURO` domina el corpus. Por decisión de diseño, el submuestreo determinista 4:1 se aplica a la unión completa **antes** de la partición. Así se preservan todos los daños y train/validación/test comparten una prevalencia controlada cercana a 20 % de daño. Esto permite una comparación interna eficiente, pero sus valores predictivos no deben interpretarse como estimaciones de producción bajo la prevalencia natural. El reponderado y remuestreo pueden ayudar, pero en clasificación multietiqueta también pueden distorsionar dependencias entre categorías (Huang et al., 2021).

### 3.5 Ausencia de fuga y selección honesta

Los dos modelos utilizarán:

- la misma instantánea humana `revision_humana_r161_cd05878518ba.json`;
- exactamente la misma partición aleatoria agrupada por video 70/15/15;
- la misma semilla, longitud, cabeza multietiqueta y criterio de parada;
- selección por **PR-AUC macro de las cinco categorías de daño en validación**;
- una sola evaluación confirmatoria en el test reservado.

La PR-AUC es la métrica principal porque el daño es raro y el accuracy puede quedar artificialmente alto prediciendo `SEGURO`. El test no se usará para escoger checkpoint, balanceo, época ni umbrales, evitando sesgo de selección (Cawley & Talbot, 2010).

## 4. Comparación cualitativa previa

| Criterio | Paraphrase MiniLM | Multilingual E5-small |
|---|---|---|
| Arquitectura | Encoder MiniLM, 12 × 384 | Encoder derivado de Multilingual MiniLM, 12 × 384 |
| Tamaño publicado | ≈0,1 B parámetros | ≈0,1 B parámetros |
| Español | Sí, multilingüe | Sí, multilingüe |
| Especialización previa | Paráfrasis y similitud de oraciones | Preentrenamiento contrastivo multilingüe y ajuste supervisado |
| Longitud publicada | 128 tokens en la configuración Sentence-Transformer | Hasta 512 tokens |
| Coste esperado a 128 tokens | Estimado: 20–30 min/época en el train 4:1 | Similar; se medirá durante la ejecución |
| Ventaja esperada | Baseline rápido y estable | Mejor transferencia semántica y opción de mayor contexto |
| Riesgo principal | Truncamiento y objetivo de paráfrasis distinto de moderación | Objetivo de recuperación distinto de moderación; exige prefijo en linear probing |

## 5. Hipótesis y regla de decisión

La hipótesis de trabajo es que E5-small puede superar al modelo de paráfrasis en PR-AUC macro de daño debido a su preentrenamiento más amplio. No existe evidencia previa específica para estas cinco categorías y español peruano que permita garantizarlo.

Primero se ejecutan seis modelos clásicos con configuraciones cerradas: los cinco ya usados en 04/04_1 más fastText supervisado OVA, encontrado en el material de la sesión 4 del profesor. No se realiza otra búsqueda de hiperparámetros. Luego se explora la pérdida con encoders congelados y se hace fine-tuning completo de cada base con la estrategia seleccionada únicamente en validación. Si E5-small obtiene mayor PR-AUC macro en validación, se lo considera ganador provisional. Su mejora sólo se considerará respaldada si en el test reservado el intervalo bootstrap pareado por video frente al mejor clásico no contradice la dirección observada y no aparece un deterioro operativo importante de recall o precisión. Si las diferencias son pequeñas o el intervalo incluye ampliamente cero, se preferirá el modelo más rápido y simple.

Para un artículo definitivo se recomienda repetir el fine-tuning con al menos tres semillas. El bootstrap por video cuantifica incertidumbre muestral, pero una sola corrida no cuantifica completamente la variabilidad de optimización.

## 6. Referencias (APA 7)

Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). *Spanish pre-trained BERT model and evaluation data*. PML4DC at ICLR 2020. https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Huang, Y., Giledereli, B., Köksal, A., Özgür, A., & Ozkirimli, E. (2021). Balancing methods for multi-label text classification with long-tailed class distribution. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 8153–8161). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.643

Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers* (pp. 427–431). Association for Computational Linguistics. https://aclanthology.org/E17-2068/

Hugging Face. (2026a). *Fine-tuning*. Transformers documentation. Recuperado el 27 de julio de 2026 de https://huggingface.co/docs/transformers/training

Hugging Face. (2026b). *intfloat/multilingual-e5-small* [Ficha de modelo]. Recuperado el 27 de julio de 2026 de https://huggingface.co/intfloat/multilingual-e5-small

Hugging Face. (2026c). *sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2* [Ficha de modelo]. Recuperado el 27 de julio de 2026 de https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). Text embeddings by weakly-supervised contrastive pre-training. *arXiv*. https://doi.org/10.48550/arXiv.2212.03533

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: A technical report. *arXiv*. https://doi.org/10.48550/arXiv.2402.05672

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained Transformers. In *Advances in Neural Information Processing Systems* (Vol. 33). https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
