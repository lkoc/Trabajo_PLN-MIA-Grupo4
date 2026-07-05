# Documento final tipo IEEE

Esta carpeta contiene el paper final del proyecto de moderacion de contenido en videos de YouTube. La estructura sigue el estilo de articulo tecnico IEEE: problema, brecha, artefacto, metodologia, evaluacion, resultados esperados y limitaciones.

## Archivos principales

- `paper_moderador_contenido_youtube_ieee.tex`: documento principal en LaTeX con clase `IEEEtran`.
- `referencias.bib`: bibliografia en BibTeX con estilo IEEE.
- `figuras/`: figuras, diagramas y fuentes graficas del paper.
- `guia_estructura_paper_ieee.md`: guia de secciones y contenido.
- `guia_redaccion_paper_ieee.md`: guia de estilo, redaccion y citas.

## Compilacion

```powershell
pdflatex -interaction=nonstopmode -halt-on-error paper_moderador_contenido_youtube_ieee.tex
bibtex paper_moderador_contenido_youtube_ieee
pdflatex -interaction=nonstopmode -halt-on-error paper_moderador_contenido_youtube_ieee.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_moderador_contenido_youtube_ieee.tex
```

## Regla editorial

El documento debe leerse como paper, no como plan de tesis. Por tanto, debe priorizar contribucion, metodo, experimento, resultados y discusion sobre formularios extensos de planificacion.
