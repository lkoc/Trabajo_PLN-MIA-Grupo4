# Huella de procedencia

## Diseño

Esta edición usa una huella **de metadatos** y no una “ciudad de papel” dentro
del corpus. No se añadieron ejemplos falsos ni se alteraron texto, etiquetas,
tiempos, pesos o particiones. Cada fila conserva:

- `release_id`: `grupo4-moderacion-youtube-peru-2026-08-17-v1.1.0`;
- `provenance_token`: HMAC-SHA256 hexadecimal de 64 caracteres;
- `provenance_key_commitment`: SHA-256 público de la clave privada.

La regla exacta es:

```text
mensaje = release_id|chunk_id|text_sha256
provenance_token = HMAC-SHA256(clave_privada, mensaje UTF-8)
```

Compromiso publicado para esta edición:

```text
SHA-256(clave_privada) = 76f0e0d5f59bda914abf64575284ff89341c4442a40c5638b95dc36cdeebef07
```

La clave privada no forma parte del ZIP ni del repositorio. El custodio debe
mantenerla en almacenamiento restringido y respaldado. Si alguna vez se revela
para una auditoría, primero se verifica que su SHA-256 coincida con el compromiso
anterior y luego se recalculan los tokens.

## Verificación por el custodio

Desde la carpeta extraída, sin copiar la clave al paquete:

```bash
python validar_dataset.py --provenance-key "/ruta/privada/publication_provenance_hmac.key"
```

Sin la clave, el mismo validador comprueba que haya 173 240 tokens
hexadecimales únicos, un compromiso uniforme y un `release_id` consistente. Con
la clave, verifica además cada HMAC.

## Qué permite afirmar

La coincidencia de `release_id`, `chunk_id`, `text_sha256` y
`provenance_token` aporta evidencia técnica de que una fila procede de esta
edición o de una copia de ella. El manifiesto y el hash del ZIP fijan además el
contenido exacto publicado.

La huella no prueba por sí sola una infracción, falta de cita ni entrenamiento
de un modelo. Puede eliminarse durante una transformación, y una fila pública
puede copiarse junto con su token. Cualquier conclusión debe apoyarse en el
contexto, las condiciones de uso y una revisión humana. La función es de
trazabilidad académica, no de vigilancia encubierta.
