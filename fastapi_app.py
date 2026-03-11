"""
APP FASTAPI - SEMANA 4
======================

Servidor asíncrono moderno que:
✔ Migra el servidor Flask a FastAPI.
✔ Añade validación estricta de tipos (Typing/Pydantic).
✔ Provee documentación automática en /docs (Swagger UI).
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
import io

from analisis_calidad_del_vino import PipelineVinos
from limpieza import eliminar_duplicados, eliminar_nulos
from schemas import VinoInputSchema, VinoOutputSchema, RespuestaLimpiezaSchema
from pydantic import ValidationError

# ─────────────────────────────────────────────
# CONFIGURACIÓN ORIGEN
# ─────────────────────────────────────────────
app = FastAPI(
    title="Análisis Enológico API",
    description="API para el análisis y limpieza de la calidad del vino mediante Pandas y Pydantic.",
    version="1.0.0"
)

RUTA_DATA = Path("data")
RUTA_DATA.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ENDPOINT 1: HOME
# ─────────────────────────────────────────────
@app.get("/", tags=["General"])
async def home() -> dict:
    return {
        "mensaje": "Servidor FastAPI - Análisis Enológico",
        "docs": "Visita /docs para ver la documentación interactiva Swagger UI."
    }


# ─────────────────────────────────────────────
# ENDPOINT 2: EJECUTAR PIPELINE COMPLETO
# ─────────────────────────────────────────────
@app.get("/analizar", tags=["Análisis"])
async def ejecutar_analisis() -> dict:
    """
    Ejecuta el pipeline de análisis completo sobre el dataset local por defecto.
    """
    try:
        pipeline = PipelineVinos("data/dataset_calidad_vinos.csv")

        (pipeline
            .ingestar()
            .eda()
            .limpiar_y_clasificar()
            .interpretar_con_ia()
        )

        return {
            "status": "ok",
            "mensaje": "Análisis ejecutado correctamente",
            "reporte_ia": pipeline.reporte_ia
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# ENDPOINT 3: LIMPIEZA CSV (SEMANA 4)
# ─────────────────────────────────────────────
@app.post("/clean", tags=["Limpieza de Datos"], response_model=RespuestaLimpiezaSchema)
async def limpiar_csv(
    file: UploadFile = File(..., description="Archivo CSV crudo a limpiar")
):
    """
    ENDPOINT DE LIMPIEZA CON VALIDACIÓN (Semana 4)
    ----------------------------------------------
    Recibe un archivo CSV mediante interfaz web (multipart/form-data), aplica 
    **validación estricta con Pydantic**, luego lógica de limpieza purista 
    (pandas pura) y guarda el resultado validado en la carpeta del servidor.
    """
    # Validación básica del formato del archivo
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV válido.")

    try:
        # 1. Leer el contenido asíncronamente y parsearlo a un DataFrame en memoria
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))

        # 2. Validación Estricta con Pydantic (Semana 4)
        # Iteramos cada fila del DataFrame original para verificar que los 
        # datos químicos sean física y biológicamente posibles según el schema.
        filas_validas = []
        for _, fila in df_raw.iterrows():
            try:
                # Si la fila rompe el contrato (ej: pH 10 o string), lanza ValidationError
                vino_validado = VinoInputSchema(**fila.to_dict())
                # Si pasó la validación, la guardamos
                filas_validas.append(vino_validado.model_dump())
            except ValidationError:
                # Las filas inválidas (basura) simplemente se ignoran silenciosamente
                pass
        
        # 3. Reconstrucción del DataFrame solo con datos 100% garantizados
        df = pd.DataFrame(filas_validas)
        if df.empty:
            raise ValueError("El CSV no contiene filas estructuralmente válidas o coherentes.")

        # 4. Limpieza básica con las funciones puras importadas de limpieza.py
        antes_duplicados = len(df)
        df = eliminar_duplicados(df)
        duplicados = antes_duplicados - len(df)

        antes_nulos = len(df)
        df = eliminar_nulos(df)
        nulos = antes_nulos - len(df)

        # 5. Exportar el DataFrame limpio y validado al disco local
        ruta_salida = RUTA_DATA / "csv_limpio_desde_fastapi.csv"
        df.to_csv(ruta_salida, index=False)

        # 6. Retorno tipeado con el esquema de Pydantic (Response Model)
        # Esto alimenta automáticamente el ejemplo de respuesta de Swagger UI
        return RespuestaLimpiezaSchema(
            status="ok",
            mensaje="Archivo validado estrictamente y limpiado correctamente",
            duplicados_eliminados=int(duplicados),
            nulos_eliminados=int(nulos),
            filas_finales=len(df),
            archivo_guardado=str(ruta_salida)
        )

    except ValueError as ve:
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")
