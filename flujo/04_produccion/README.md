# Etapa 04 · Frontend de producción supervisada

El cuaderno `04_01_frontend_produccion.ipynb` comprueba que exista un registro de modelos del contrato v2.1 e inicia el frontend local:

```powershell
modperu serve-production --host 127.0.0.1 --port 8765
```

La interfaz muestra cinco scores aprendidos, contrato, modelo y motivos de revisión. Un conflicto `SEGURO+daño`, ninguna salida sobre umbral o cercanía a un umbral obliga revisión humana. Si falta el registro v2.1, el servidor no carga un modelo histórico como sustituto.
