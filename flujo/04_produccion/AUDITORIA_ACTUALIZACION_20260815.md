# Auditoría de actualización de producción · 15 de agosto de 2026

## Resultado

El frontend y el servidor quedaron alineados con la selección congelada por `03_07` y fueron verificados con inferencia real local sobre CPU:

- clásico: `classical-logistic_regression_c0p5-54f7971c6000`;
- Transformer: `cascade_v2-af78eba77883`;
- Qwen: `qwen_lora-4aa5ce04df05`;
- ensemble: `ensemble_soft_mean`.

El ensemble reproduce el orden congelado: promedio de los *scores* crudos de sus tres miembros, calibración sigmoidal por salida, umbrales por categoría, compuerta binaria de daño y política `NEEDS_REVIEW`. El registro principal permanece en `shadow_only` y conserva `winner_status=statistical_tie_or_inconclusive`.

## Restauración verificable del Transformer

El archivo descargado por el navegador se copió desde `Downloads` a:

```text
modelos/_downloads_04/transformer_03_03b_run_outputs.tar.gz
```

Controles aplicados:

- tamaño exacto: 2 854 627 749 bytes;
- SHA-256: `fff9f75ae381ec0123b57850afc72528bd27e4d6ae75b8e3a6aedf150bbab290`;
- candidato extraído: `runs/cascade_v2-af78eba77883921f`;
- destino local: `modelos/v2/transformers/production/cascade_v2-af78eba77883921f`;
- 16 archivos finales, 985 653 357 bytes;
- los 13 archivos declarados por el manifiesto del checkpoint coincidieron individualmente.

No se extrajeron checkpoints intermedios del entrenador.

## Registros y modos publicados

Se materializaron los cuatro registros locales:

```text
modelos/registro_modelos_5_salidas.json
modelos/registro_modelos_5_salidas.classical.json
modelos/registro_modelos_5_salidas.transformer.json
modelos/registro_modelos_5_salidas.qwen.json
```

`GET /api/config` confirmó los modos `classical`, `transformer`, `qwen`, `compare` y `ensemble`; el modo predeterminado es `ensemble`. La opción `compare` devuelve los tres modelos individuales y el ensemble en una misma respuesta.

## Prueba funcional local

Se inició el servidor de producción en `http://127.0.0.1:8876` y se enviaron solicitudes reales a la API:

- el clásico cargó y produjo scores de las cinco salidas;
- la cascada cargó compuerta y rama y produjo scores de las cinco salidas;
- Qwen cargó el adaptador de 22 logits y produjo las cinco salidas primarias;
- el ensemble devolvió exactamente los tres miembros congelados y su promedio calibrado;
- la comparación devolvió cuatro paneles: clásico, Transformer, Qwen y ensemble.

La primera carga de Qwen requirió resolver la cadena de certificados corporativa y completar la caché local de `Qwen/Qwen3-0.6B-Base`, revisión `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Después de ello, la inferencia en CPU terminó correctamente. El aviso de cabeza base no inicializada aparece antes de cargar el adaptador PEFT y no impidió recuperar la cabeza entrenada.

## Controles incorporados

- Materialización automática desde `resultados/modelos/seleccion_congelada.json`.
- Verificación de identificadores, contrato taxonómico, hashes, manifiestos y archivos requeridos.
- Selector de modelo individual, comparación o ensemble.
- Carga flexible de las cinco salidas primarias y 17 salidas auxiliares del adaptador Qwen.
- Inclusión portable de `branch_model` en la cascada v2.
- Persistencia separada por modelo, categoría y estado de revisión.

La suite completa del proyecto finalizó con 222 pruebas aprobadas.

## Límite de versionado

Los pesos, la caché de Hugging Face, el archivo de 2,85 GB y los eventos de la prueba local permanecen fuera de Git. Git conserva el código, los cuadernos, las pruebas, la documentación y las instrucciones reproducibles para volver a materializar los registros.
