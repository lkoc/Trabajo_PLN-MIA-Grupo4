# Cuadernos del proyecto

Esta carpeta organiza el flujo reproducible del moderador de contenido para videos publicos de YouTube en espanol peruano. La secuencia es: recoleccion, limpieza, etiquetado, entrenamiento e inferencia.

## Orden de ejecucion

1. `01_scraping_youtube_politica_farandula.ipynb`: descubre canales candidatos, registra metadatos y descarga transcripciones publicas cuando esten disponibles.
   - `01_1_ampliacion_dirigida_dano.ipynb`: orquesta el lote aislado de ampliacion dirigida, desde subtitulos y chunks hasta Flash, Pro, balance y reentrenamiento condicionado a la adjudicacion humana.
2. `02_limpieza_y_chunks.ipynb`: normaliza transcripciones, crea fragmentos etiquetables y elimina duplicados.
3. `03_frontend_etiquetado_humano_html.ipynb`: prepara el frontend local de etiquetado humano y valida el formato de salida.
4. Entrenamiento activo con cuatro daños (`ACOSO_PERSONAL` y `AMENAZA_DIRECTA` se integran como `ACOSO_AMENAZA`). El contrato de datos, la función de cada cuaderno y el régimen de etiquetas auxiliares están en `04_MATRIZ_ENTRENAMIENTO_4_ETIQUETAS.md`.
   - `04_200_ORDEN_EJECUCION.md`: índice y dependencias de la fase de entrenamiento.
   - `04_201_clasicos_planos_y_jerarquicos_4_etiquetas.ipynb`: SVM y logística planos, en cascada y jerárquicos.
   - `04_202_transformers_planos_4_etiquetas.ipynb`: MiniLM y E5 planos.
   - `04_203_transformer_cascada_4_etiquetas.ipynb`: Transformer en dos etapas.
   - `04_204_transformer_jerarquico_multitarea_4_etiquetas.ipynb`: Transformer jerárquico con encoder compartido.
   - `04_205_finetuning_qwen_acoso_amenaza.ipynb`: Qwen3-0.6B LoRA plano con supervisión auxiliar fina/transversal; completó cuatro épocas y seleccionó la época operativa 3 sólo con validación.
   - `04_206_qwen_cascada_y_jerarquico_4_etiquetas.ipynb`: cabezas de cascada y multitarea sobre representaciones de la época operativa 3; reutiliza entrenamientos existentes y sincroniza `04_205` desde Drive si hace falta.
   - `04_207_comparacion_final_modelos_4_etiquetas.ipynb`: comparación sobre validation/test 4:1 comunes; recupera resultados remotos faltantes y evita duplicar la referencia Qwen plana.
   - `04_208_auditoria_finas_transversales_modelos_4.ipynb`: auditoría común por etiquetas finas, flags e incertidumbre; verifica Drive y registra la procedencia del checkpoint Qwen.
   - `04_old_5etiquetas/`: cuadernos, resultados y modelos históricos de cinco etiquetas, conservados para reproducibilidad y warm start.
5. `05_frontend_produccion.ipynb`: inicia el servidor local con HTML autocontenido. Detecta texto o YouTube, rechaza videos sin subtítulos, permite escoger/ comparar/consensuar los tres mejores tipos de modelo y guarda revisión humana y estadísticas. La guía está en `05_MODO_OPERACION.md`.

## Directorios de salida

- `datos/raw`: metadatos y transcripciones originales.
- `datos/interim`: transcripciones limpias y fragmentos intermedios.
- `datos/processed`: chunks etiquetables y dataset etiquetado.
- `modelos`: modelos entrenados, vectorizadores y etiquetas.
- `resultados`: metricas, auditorias y exportaciones.
- `Cuadernos/frontend`: frontends HTML locales.

## Criterio de uso de datos

El proyecto usa contenido público, respeta los términos de servicio y conserva sólo los datos necesarios para investigación. La lista de canales es una semilla editable; cada fuente debe verificarse antes de la descarga. Las credenciales nunca se guardan en los cuadernos. Si se usa un servicio externo autorizado para una fase de etiquetado, su salida, coste y procedencia se registran y la decisión final sigue sujeta a revisión humana.
