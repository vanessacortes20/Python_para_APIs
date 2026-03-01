"""
APP FLASK - SEMANA 3
====================

Mini servidor web síncrono que:
- Ejecuta el análisis completo
- Permite limpiar datos vía endpoint
- Devuelve resultados en JSON

Demuestra:
✔ Request / Response
✔ Endpoints GET y POST
✔ Integración con tu pipeline
"""

from flask import Flask, jsonify, request
from analisis import PipelineVinos
import pandas as pd
from pathlib import Path

app = Flask(__name__)

RUTA_DATA = Path("Data")
RUTA_DATA.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ENDPOINT 1: Home
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
# ENDPOINT 2: Ejecutar análisis completo
# ─────────────────────────────────────────────
@app.route("/analizar", methods=["GET"])
def ejecutar_analisis():
    try:
        pipeline = PipelineVinos("winequality-red.csv")
        pipeline \
            .cargar_datos() \
            .limpiar_datos() \
            .clasificar_vinos() \
            .analisis_exploratorio() \
            .interpretar_con_ia()

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
# ENDPOINT 3: Limpieza de datos (POST)
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
        df = df.drop_duplicates()
        df = df.dropna()

        ruta_salida = RUTA_DATA / "csv_limpio_desde_api.csv"
        df.to_csv(ruta_salida, index=False)

        return jsonify({
            "status": "ok",
            "mensaje": "Archivo limpiado correctamente",
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