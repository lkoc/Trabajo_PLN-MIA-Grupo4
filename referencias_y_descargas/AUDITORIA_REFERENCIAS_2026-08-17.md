# Auditoría final de citas, referencias y documentos

Fecha de cierre: 17 de agosto de 2026

Alcance: versión compilable de `Documento_final_paper`, incluidas sus secciones
y anexos. El inventario reproducible se conserva en
`indice_referencias.csv`.

## Resultado ejecutivo

- 101 claves distintas están citadas en el manuscrito y las 101 tienen una
  entrada BibTeX; no hay claves duplicadas ni citas indefinidas.
- No se detectaron referencias ficticias, títulos atribuidos a otra obra ni DOI
  inventados.
- 65 referencias poseen DOI. Los 65 DOI resolvieron y sus metadatos de título y
  autor coinciden con la referencia: 59 se contrastaron en Crossref y seis DOI
  de arXiv u otros repositorios se verificaron mediante el resolvedor DOI y sus
  metadatos CSL/DataCite.
- Las 36 fuentes restantes no tienen un DOI asignado en su publicación
  canónica. De ellas, 19 cuentan con PDF local y 17 son documentación, normas,
  fichas de modelo, catálogos o páginas oficiales sin PDF.
- 72 de las 101 referencias cuentan con PDF local identificado. Los 72
  documentos corresponden a su cita: 71 títulos se comprobaron por extracción
  de texto y el informe escaneado de Chow (1970) se confirmó visualmente.
- La carpeta contiene otros 13 PDF de entradas bibliográficas históricas que ya
  no se citan. Se conservan, pero el índice vigente no los contabiliza como
  evidencia del artículo.

## Procedimiento de comprobación

1. Se extrajeron todas las claves de `\cite`, `\citep`, `\citet` y variantes
   desde los archivos TeX efectivamente incluidos por el documento principal.
2. Las claves se cruzaron con `referencias.bib`; también se buscaron claves
   repetidas y referencias no definidas.
3. Cada DOI se resolvió contra infraestructura DOI y se cotejaron título,
   primer autor y año con los metadatos del registro. Dos diferencias de año
   son legítimas: Chakravarthi et al. figura en línea en 2023 y en el volumen de
   2024; Silla y Freitas apareció en línea en 2010 y en el volumen de 2011. El
   BibTeX usa el año del volumen.
4. Para las fuentes sin DOI se consultó la página primaria del editor,
   repositorio, organismo, catálogo o proyecto. La ausencia de DOI no se trató
   como ausencia de la fuente.
5. Cada PDF local se validó por firma `%PDF`, lectura estructural y
   correspondencia de título/autor. No se aceptaron como prueba archivos con
   una mera extensión PDF.
6. Se revisaron las afirmaciones del manuscrito para comprobar que los métodos,
   antecedentes, definiciones y comparaciones externas tuvieran una atribución
   próxima. Los resultados producidos por el proyecto se mantienen como
   elaboración propia.

Fuentes primarias de control: [documentación de la API REST de
Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/),
[catálogo del Fondo Editorial PUCP para *Racismo y
lenguaje*](https://www.fondoeditorial.pucp.edu.pe/linguistica/525-racismo-y-lenguaje.html),
[catálogo bibliotecario de
Callirgos](https://biblioteca.unap.edu.pe/opac_css/index.php?lvl=notice_display&id=8721),
[catálogo del Fondo Editorial del
Congreso](https://www2.congreso.gob.pe/Sicr/FondoEditorial/SIFonEdi.nsf/Catalogomateria?OpenForm)
y [registro de Dialnet del volumen de van
Dijk](https://dialnet.unirioja.es/servlet/libro?codigo=270993).

## Correcciones realizadas

- Se normalizaron los 65 enlaces DOI al formato canónico
  `https://doi.org/<doi>` para que aparezcan en la bibliografía compilada.
- Se corrigió el ISBN de *Racismo y lenguaje* a `978-612-317-255-8` y se enlazó
  su ficha editorial oficial.
- Se añadieron o sustituyeron catálogos verificables para Callirgos (1993),
  Portocarrero (2009) y el capítulo de Zavala y Zariquiey (2007).
- En el anexo metodológico se añadieron las fuentes de Platt y de
  Niculescu-Mizil y Caruana a la explicación de calibración de probabilidades.
- Se regeneró `indice_referencias.csv`; ahora contiene exactamente una fila por
  referencia citada en el artículo.

## Documentos incorporados en esta auditoría

| Clave | Documento local | Procedencia comprobada |
|---|---|---|
| `chow1970reject` | `chow1970reject__chow_1970_optimum-reject-tradeoff.pdf` | [Informe técnico conservado por MIT DSpace](https://hdl.handle.net/1721.1/6177), correspondiente al trabajo publicado. |
| `cortes1995svm` | `cortes1995svm__cortes_vapnik_1995_support-vector-networks.pdf` | [Copia del editor Springer](https://doi.org/10.1007/BF00994018). |
| `field2007clusterbootstrap` | `field2007clusterbootstrap__field_welsh_2007_cluster-bootstrap.pdf` | Copia académica cotejada con la [ficha del editor y su DOI](https://doi.org/10.1111/j.1467-9868.2007.00593.x). |
| `thakur2025quechua` | `thakur2025quechua__thakur_2025_moderacion-quechua.pdf` | [Informe oficial del Center for Democracy & Technology](https://cdt.org/insights/moderating-quechua-content-on-social-media/). |

No se forzaron descargas detrás de barreras de pago ni se eludieron controles
antibot. Por ejemplo, la página pública del manuscrito de *Super Learner* fue
localizada, pero el servidor rechazó la descarga automatizada con HTTP 403; el
estado queda explícito en el CSV. Una falta de PDF local significa solamente
que no se recuperó una copia abierta adecuada, no que la referencia sea
inexistente.

## Auditoría del citado en el texto

La revisión no encontró pasajes que reproduzcan material ajeno sin atribución.
Las afirmaciones externas sobre arquitecturas, calibración, métricas,
moderación, taxonomías, ética y antecedentes cuentan con una cita próxima. Las
tablas y figuras identifican su fuente o indican «elaboración propia», según
corresponde. Los valores de los experimentos, las decisiones del flujo y las
fórmulas específicas de los ensembles se presentan como resultados o
propuestas del trabajo y no se atribuyen a literatura inexistente.

Esta auditoría reduce el riesgo bibliográfico y deja evidencia comprobable,
pero no sustituye una evaluación institucional de similitud textual ni una
determinación jurídica de plagio.
