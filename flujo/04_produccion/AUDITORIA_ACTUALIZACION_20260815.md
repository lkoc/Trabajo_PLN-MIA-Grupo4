# Auditoría de actualización de producción · 15 de agosto de 2026

## Alcance implementado

El frontend y el servidor quedaron alineados con la selección congelada por `03_07`:

- clásico: `classical-logistic_regression_c0p5-54f7971c6000`;
- Transformer: `cascade_v2-af78eba77883`;
- Qwen: `qwen_lora-4aa5ce04df05`;
- ensemble: `ensemble_soft_mean`.

El ensemble productivo reproduce el orden congelado: promedio de los *scores* crudos de sus tres miembros, calibración sigmoidal por salida, umbrales por categoría, compuerta binaria de daño y política `NEEDS_REVIEW`. El registro principal se publica como `shadow_only` y conserva `winner_status=statistical_tie_or_inconclusive`.

## Controles incorporados

- Materialización automática del registro principal y los tres registros miembro a partir de `resultados/modelos/seleccion_congelada.json`.
- Verificación de identificadores, contrato taxonómico, SHA-256 del *snapshot*, manifiestos y archivos de los *checkpoints*.
- Selector para ejecutar el ensemble, cada modelo por separado o la comparación de los cuatro sistemas.
- Carga flexible de Qwen con las 5 salidas primarias y las salidas auxiliares conservadas por el adaptador.
- Inclusión de `branch_model` en los registros portables de la cascada v2.
- Persistencia separada por modelo, categorías y estado de revisión.

La suite completa del proyecto finalizó con 222 pruebas aprobadas.

## Estado de los pesos locales

- Clásico: disponible y verificable.
- Qwen LoRA: disponible; el archivo restaurado desde Drive coincide con el SHA-256 publicado.
- Transformer: pendiente de restauración local.

No se genera `modelos/registro_modelos_5_salidas.json` mientras falte un miembro. Esta detención evita sustituir silenciosamente el ganador por un modelo histórico o publicar un ensemble incompleto.

## Bloqueo externo restante

Drive contiene el Transformer dentro de `run_outputs.tar.gz`, archivo de 2,854,627,749 bytes con SHA-256:

```text
fff9f75ae381ec0123b57850afc72528bd27e4d6ae75b8e3a6aedf150bbab290
```

El conector disponible limita cada descarga a 268,435,456 bytes y rechaza el archivo monolítico. Para concluir la prueba integrada debe descargarse desde el navegador y guardarse como:

```text
C:\usr\ths_mia_fiis\pln\trabajo\modelos\_downloads_04\transformer_03_03b_run_outputs.tar.gz
```

Después se verificará su hash, se extraerá únicamente `cascade_v2-af78eba77883921f`, se publicarán los cuatro registros y se ejecutará una inferencia real desde el frontend.
