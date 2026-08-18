# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Referencias y descargas

Esta carpeta reúne las fuentes citadas de forma efectiva en el artículo. El
inventario se auditó y cerró el 17 de agosto de 2026 contra
`Documento_final_paper/secciones/*.tex` y
`Documento_final_paper/referencias.bib`.

## Resumen del inventario

- Fuentes citadas e indexadas: 101.
- Referencias con DOI verificado: 65.
- Referencias sin DOI asignado: 36.
- PDF locales vinculados a referencias vigentes: 72 (65,95 MiB).
- Fuentes vigentes sin PDF local: 29.
- PDF históricos de entradas que ya no se citan: 13.
- Claves citadas sin entrada BibTeX: 0.
- PDF cuya firma inicial no sea `%PDF`: 0.

El archivo `indice_referencias.csv` contiene una fila por clave citada y las
columnas `clave`, `titulo`, `tipo`, `doi`, `url`, `pdf_local` y `estado`. Los
estados distinguen entre un PDF validado, una fuente web sin versión PDF, una
obra impresa o editorial sin copia abierta localizada y una descarga abierta
bloqueada por el servidor. Así se evita confundir «no descargado» con «fuente
inexistente».

El informe `AUDITORIA_REFERENCIAS_2026-08-17.md` documenta el contraste de
existencia, identidad, DOI, metadatos y correspondencia de los PDF. Los 13 PDF
históricos se conservan porque pertenecen al catálogo bibliográfico del
proyecto, pero no se contabilizan como respaldo de las 101 referencias que
aparecen en la versión vigente del artículo.

| Patrón de `estado` | Lectura práctica |
|---|---|
| `pdf_oa_validado` | Existe una copia local y superó la validación binaria. |
| `*_web_oficial_sin_pdf` o `*_web_sin_pdf` | La fuente canónica es una página web. |
| `fuente_impresa_*` o `fuente_editorial_*` | La referencia existe, pero no se localizó una copia PDF abierta adecuada. |
| `*_bloqueada_403` o `*_bloqueada_antibot` | Se identificó un PDF público, pero el servidor impidió la descarga automatizada. |
| `repositorio_oficial_*` | El repositorio conserva metadatos, pero no ofrece un PDF descargable en las condiciones observadas. |

## Criterio de descarga

Solo se conservan PDF obtenidos de editores, autores o repositorios oficiales,
entre ellos ACL Anthology, PMLR, JMLR, arXiv, NeurIPS, NIST, repositorios
universitarios y revistas de acceso abierto. No se incluyen copias de
procedencia dudosa ni se eluden barreras de pago. Cuando una norma, ficha de
modelo, documentación de software o condición de servicio solo existe como
recurso web, el índice conserva su URL oficial sin fabricar una conversión a
PDF.

Cada archivo sigue la convención
`clave_bibtex__autor_anio_titulo-corto.pdf`. Todos los PDF locales se validaron
por su firma binaria inicial `%PDF` y por lectura de metadatos. En los 72 PDF
vinculados también se cotejaron título y autor con la entrada BibTeX: 71
superaron la comprobación textual automatizada y el informe escaneado de Chow
(1970), sin capa de texto, se confirmó mediante inspección visual. Una
extensión `.pdf` por sí sola no se consideró suficiente.

## Actualización reproducible

`generar_indice_referencias.ps1` vuelve a leer las citas del artículo, cruza
las claves con la bibliografía y detecta los PDF presentes. El script imprime
el CSV en la salida estándar y, si se indica `-OutputPath`, también lo guarda;
no descarga ni modifica las fuentes:

```powershell
pwsh -NoProfile -File .\referencias_y_descargas\generar_indice_referencias.ps1 `
  -OutputPath .\referencias_y_descargas\indice_referencias.csv
```

Si después se agregan o quitan citas, debe regenerarse
`indice_referencias.csv` y validarse de nuevo el conjunto de PDF. Los estados
de descarga bloqueada también deben revisarse porque los repositorios pueden
cambiar sus rutas o políticas.

## Derechos

Los derechos de cada documento pertenecen a sus autores y editores. La copia
local se guarda para consulta académica y reproducibilidad; su presencia en el
proyecto no cambia la licencia original ni autoriza una redistribución distinta
de la permitida por la fuente. Antes de publicar esta carpeta en un repositorio
externo debe comprobarse la licencia de cada archivo.
