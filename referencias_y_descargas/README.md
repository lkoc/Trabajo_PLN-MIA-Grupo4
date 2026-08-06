# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Referencias y descargas

Esta carpeta reúne las fuentes citadas de forma efectiva en el artículo. El
inventario se cerró el 29 de julio de 2026 contra
`Documento_final_paper/secciones/*.tex` y
`Documento_final_paper/referencias.bib`.

## Resumen del inventario

- Fuentes citadas e indexadas: 117.
- PDF locales válidos: 81 (76,21 MiB).
- Fuentes sin PDF local: 36.
- Claves citadas sin entrada BibTeX: 0.
- PDF cuya firma inicial no sea `%PDF`: 0.

El archivo `indice_referencias.csv` contiene una fila por clave citada y las
columnas `clave`, `titulo`, `tipo`, `doi`, `url`, `pdf_local` y `estado`. Los
estados distinguen entre un PDF validado, una fuente web sin versión PDF, una
obra impresa o editorial sin copia abierta localizada y una descarga abierta
bloqueada por el servidor. Así se evita confundir «no descargado» con «fuente
inexistente».

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
por su firma binaria inicial `%PDF`; una extensión `.pdf` por sí sola no se
consideró suficiente.

## Actualización reproducible

`generar_indice_referencias.ps1` vuelve a leer las citas del artículo, cruza
las claves con la bibliografía y detecta los PDF presentes. El script imprime
el CSV actualizado en la salida estándar y no descarga ni modifica fuentes:

```powershell
pwsh -NoProfile -File .\referencias_y_descargas\generar_indice_referencias.ps1
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
