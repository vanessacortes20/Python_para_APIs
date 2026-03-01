"""
APP FLASK - SEMANA 3
====================

Mini servidor web síncrono que:

✔ Ejecuta el pipeline completo de análisis
✔ Permite limpiar un CSV vía endpoint POST
✔ Devuelve resultados en formato JSON

Cumple con:
- Request / Response
- Endpoints GET y POST
- Migración de limpieza a /clean
"""

from flask import Flask, jsonify, request
from analisis_calidad_del_vino import PipelineVinos
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
app = Flask(__name__)

RUTA_DATA = Path("data")
RUTA_DATA.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ENDPOINT 1: HOME
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "Servidor Flask - Análisis Enológico",
        "endpoints": {
            "GET /analizar": "Ejecuta el pipeline completo",
            "POST /clean": "Limpia un CSV enviado"
        }
    })


# ─────────────────────────────────────────────
# ENDPOINT 2: EJECUTAR PIPELINE COMPLETO
# ─────────────────────────────────────────────
@app.route("/analizar", methods=["GET"])
def ejecutar_analisis():
    try:
        pipeline = PipelineVinos("data/dataset_calidad_vinos.csv")

        (pipeline
            .ingestar()
            .eda()
            .limpiar_y_clasificar()
            .interpretar_con_ia()
        )

        return jsonify({
            "status": "ok",
            "mensaje": "Análisis ejecutado correctamente",
            "reporte_ia": pipeline.reporte_ia
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "detalle": str(e)
        }), 500


# ─────────────────────────────────────────────
# ENDPOINT 3: LIMPIEZA CSV (POST)
# ─────────────────────────────────────────────
@app.route("/clean", methods=["POST"])
def limpiar_csv():

    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    try:
        df = pd.read_csv(file)

        # Limpieza básica
        duplicados = df.duplicated().sum()
        df = df.drop_duplicates()

        nulos = df.isna().sum().sum()
        df = df.dropna()

        ruta_salida = RUTA_DATA / "csv_limpio_desde_api.csv"
        df.to_csv(ruta_salida, index=False)

        return jsonify({
            "status": "ok",
            "mensaje": "Archivo limpiado correctamente",
            "duplicados_eliminados": int(duplicados),
            "nulos_eliminados": int(nulos),
            "filas_finales": len(df),
            "archivo_guardado": str(ruta_salida)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "detalle": str(e)
        }), 500


# ─────────────────────────────────────────────
# EJECUCIÓN DEL SERVIDOR
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)