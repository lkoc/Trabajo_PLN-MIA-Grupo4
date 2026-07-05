# Moderador de Contenido en Videos de YouTube

Trabajo final del curso de Procesamiento de Lenguaje Natural, Maestria en Inteligencia Artificial, Universidad Nacional de Ingenieria. Grupo 4, semestre 2026-1.

## Objetivo

Construir un flujo local y reproducible para recolectar texto de videos publicos de YouTube, crear fragmentos auditables, etiquetarlos con revision humana, entrenar un clasificador de moderacion de contenido y presentar los resultados en formato de paper IEEE.

El proyecto se enfoca en contenido peruano de politica, periodismo, farandula, humor, streaming y viajes. La recoleccion y el procesamiento se realizan sin APIs externas.

## Estructura del repositorio

```text
.
|-- Cuadernos/
|   |-- 01_scraping_youtube_politica_farandula.ipynb
|   |-- 02_limpieza_y_chunks.ipynb
|   |-- 03_frontend_etiquetado_humano_html.ipynb
|   |-- 04_entrenamiento_moderador.ipynb
|   |-- 05_frontend_produccion.ipynb
|   `-- frontend/
|-- Presentación_BEAMER/
|   |-- presentacion_grupo4.tex
|   `-- presentacion_grupo4.pdf
|-- Documento_final_paper/
|   |-- paper_moderador_contenido_youtube_ieee.tex
|   |-- referencias.bib
|   |-- figuras/
|   |-- guia_estructura_paper_ieee.md
|   `-- guia_redaccion_paper_ieee.md
|-- bibliografia/
|-- scripts_auxiliares/
|-- datos/
|-- modelos/
|-- resultados/
`-- PLN_clases/
```

## Flujo de trabajo

1. Revisar los canales candidatos en `bibliografia/canales_candidatos.md`.
2. Ejecutar los cuadernos en orden desde `Cuadernos/`.
3. Guardar datos crudos en `datos/raw`, intermedios en `datos/interim` y datasets finales en `datos/processed`.
4. Guardar modelos entrenados en `modelos/`.
5. Guardar metricas, figuras y reportes en `resultados/`.
6. Actualizar el paper IEEE en `Documento_final_paper/`.
7. Actualizar la presentacion Beamer en `Presentación_BEAMER/`.

## Compilacion

Beamer:

```powershell
cd "Presentación_BEAMER"
pdflatex -interaction=nonstopmode -halt-on-error presentacion_grupo4.tex
```

Paper IEEE:

```powershell
cd Documento_final_paper
pdflatex -interaction=nonstopmode -halt-on-error paper_moderador_contenido_youtube_ieee.tex
bibtex paper_moderador_contenido_youtube_ieee
pdflatex -interaction=nonstopmode -halt-on-error paper_moderador_contenido_youtube_ieee.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_moderador_contenido_youtube_ieee.tex
```

## Criterios del proyecto

- No usar APIs para recolectar datos, etiquetar contenido ni clasificar texto.
- Mantener evidencia trazable: video, canal, fragmento, categoria, score y texto activador.
- Usar revision humana para las etiquetas y para los casos ambiguos.
- Priorizar un baseline local defendible antes de modelos Transformer.
- Evitar datos innecesarios, duplicados o no auditables.
