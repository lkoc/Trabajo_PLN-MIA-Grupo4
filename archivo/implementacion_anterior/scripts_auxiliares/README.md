# Scripts auxiliares

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Esta carpeta contiene el código reproducible que comparten los cuadernos: preparación de datos, entrenamiento, calibración, evaluación, comparación, auditoría y sincronización de artefactos.

## Módulos activos de la fase 04

- `entrenar_qwen_acoso_amenaza.py`: fine-tuning Qwen, reanudación, calibración y cargador canónico de la selección operativa.
- `experimentos_qwen_jerarquico_4.py`: extracción de representaciones y cabezas Qwen en cascada/multitarea; puede recalcular comparaciones sin reentrenar.
- `analizar_auxiliares_modelos_4.py`: auditoría común de etiquetas finas, flags e incertidumbre.
- `recuperar_resultados_colab_04_20x.ps1`: recuperación Drive → `D:` con SHA-256, alcances `Qwen04_205Only`, `Qwen04_206Only` y `Qwen04_20XOnly`, y política opcional `PreferNewest` usada por 04_208.
- `sincronizar_04_20x_google_drive.ps1`: preparación del bundle persistente para Colab.
- `registro_modelos_produccion_4.py`: publica los mejores modelos clásico, MiniLM y Qwen seleccionados sólo con validation.
- `servidor_moderacion_05.py`: inferencia local, subtítulos YouTube, consenso, revisión, SQLite y exportación deduplicada para reentrenamiento.
- `crear_bundle_despliegue_05.py`: empaqueta aplicación, modelos, base operativa
  vacía, manifiesto, Docker y guías en `05_frontend_despliegue/`.
- `generar_anexo_d_canales.py`: cruza el corpus integrado con los metadatos de
  adquisición, deduplica por `channel_id` y regenera la tabla multipágina de
  canales y URL canónicas del Apéndice D.

Los scripts no deben contener credenciales. Todo artefacto consumido por otro cuaderno debe conservar ruta, hash, dataset y partición usada para selección.
