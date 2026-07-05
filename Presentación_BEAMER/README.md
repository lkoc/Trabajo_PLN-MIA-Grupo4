# Presentacion Beamer

Esta carpeta contiene la presentacion del Grupo 4 para el trabajo final del curso de Procesamiento de Lenguaje Natural.

## Archivos

- `presentacion_grupo4.tex`: fuente LaTeX Beamer.
- `presentacion_grupo4.pdf`: PDF compilado.
- `Moderador_Contenido_YouTube_PLN.pptx`: exportacion o version PPTX del tema actual, si se requiere para entrega.

## Compilacion

```powershell
pdflatex -interaction=nonstopmode -halt-on-error presentacion_grupo4.tex
```

La presentacion debe mantenerse alineada con `Cuadernos/01_scraping_youtube_politica_farandula.ipynb` y con el paper IEEE.
