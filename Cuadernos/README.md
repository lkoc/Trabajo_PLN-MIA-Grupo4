# Cuadernos del proyecto

Esta carpeta organiza el flujo reproducible del moderador de contenido para videos publicos de YouTube en espanol peruano. La secuencia es: recoleccion, limpieza, etiquetado, entrenamiento e inferencia.

## Orden de ejecucion

1. `01_scraping_youtube_politica_farandula.ipynb`: descubre canales candidatos, registra metadatos y descarga transcripciones publicas cuando esten disponibles.
2. `02_limpieza_y_chunks.ipynb`: normaliza transcripciones, crea fragmentos etiquetables y elimina duplicados.
3. `03_frontend_etiquetado_humano_html.ipynb`: prepara el frontend local de etiquetado humano y valida el formato de salida.
4. `04_entrenamiento_moderador.ipynb`: entrena un baseline clasico y deja preparado el ajuste fino con modelos Transformer.
5. `05_frontend_produccion.ipynb`: arma el prototipo de inferencia local y el frontend de produccion.

## Directorios de salida

- `datos/raw`: metadatos y transcripciones originales.
- `datos/interim`: transcripciones limpias y fragmentos intermedios.
- `datos/processed`: chunks etiquetables y dataset etiquetado.
- `modelos`: modelos entrenados, vectorizadores y etiquetas.
- `resultados`: metricas, auditorias y exportaciones.
- `Cuadernos/frontend`: frontends HTML locales.

## Criterio de uso de datos

El proyecto usa contenido publico, respeta terminos de servicio y conserva solo los datos necesarios para investigacion. La lista de canales es una semilla editable; cada fuente debe verificarse manualmente antes de entrar a descarga automatica. No se usan llaves ni APIs externas.
