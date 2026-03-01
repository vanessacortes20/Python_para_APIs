"""
╔══════════════════════════════════════════════════════════════╗
║ SISTEMA INTEGRADO DE ANÁLISIS ENOLÓGICO                     ║
║ Semanas 1, 2 y 3                                             ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
# 1. LIBRERÍAS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from decorators import registrar_ejecucion, validar_normalidad
from schemas import VinoSchema


# ─────────────────────────────────────────────
# 2. CONFIGURACIÓN
# ─────────────────────────────────────────────
RUTA_OUTPUTS = Path("outputs")
RUTA_OUTPUTS.mkdir(exist_ok=True)

RUTA_DATA = Path("data")
RUTA_DATA.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True
})


# ─────────────────────────────────────────────
# 3. SEMANA 1 - PATTERN MATCHING
# ─────────────────────────────────────────────
def clasificar_vino(data: dict) -> str:
    match data:
        case {"calidad": c} if c >= 8:
            return "premium"
        case {"calidad": c} if c >= 6:
            return "estandar"
        case {"calidad": c} if c >= 4:
            return "economico"
        case _:
            return "baja_calidad"


# ─────────────────────────────────────────────
# 4. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
class PipelineVinos:

    def __init__(self, ruta_csv: str):
        self.ruta = ruta_csv
        self.df = None
        self.reporte_eda = {}
        self.reporte_ia = {}

    # ─────────────────────────────
    # INGESTA + VALIDACIÓN (Semana 2)
    # ─────────────────────────────
    @registrar_ejecucion
    def ingestar(self):
        raw_df = pd.read_csv(self.ruta)
        validos = []

        for _, fila in raw_df.iterrows():
            try:
                v = VinoSchema(**fila.to_dict())
                validos.append(v.model_dump())
            except:
                pass

        self.df = pd.DataFrame(validos)
        print(f"✓ Registros válidos: {len(self.df)}")
        return self

    # ─────────────────────────────
    # EDA COMPLETO
    # ─────────────────────────────
    @registrar_ejecucion
    @validar_normalidad(alpha=0.05)
    def eda(self):

        print("\n═══ INICIANDO EDA COMPLETO ═══")

        # Información general
        info = {
            "filas": len(self.df),
            "columnas": len(self.df.columns),
            "duplicados": int(self.df.duplicated().sum())
        }

        # Nulos
        nulos = self.df.isna().sum().to_dict()

        # Estadísticas descriptivas
        descriptivo = self.df.describe().to_dict()

        # Outliers por IQR
        outliers = {}
        for col in self.df.select_dtypes(include="number").columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lim_sup = q3 + 1.5 * iqr
            lim_inf = q1 - 1.5 * iqr
            cantidad = len(self.df[(self.df[col] > lim_sup) | (self.df[col] < lim_inf)])
            outliers[col] = cantidad

        # Correlación
        correlacion = self.df.corr(numeric_only=True).to_dict()

        # Guardar reporte EDA
        self.reporte_eda = {
            "info_general": info,
            "nulos": nulos,
            "estadisticas": descriptivo,
            "outliers_detectados": outliers,
            "correlacion": correlacion,
            "decisiones_limpieza": {
                "duplicados": "Eliminados para evitar sobre-representación de muestras.",
                "nulos": "Eliminados para preservar integridad química real.",
                "outliers": "Se excluye percentil 98 en alcohol para estabilidad estadística."
            }
        }

        with open(RUTA_OUTPUTS / "reporte_eda.json", "w") as f:
            json.dump(self.reporte_eda, f, indent=4)

        # ───────── IMÁGENES ─────────

        # 1. Distribuciones
        fig, axes = plt.subplots(2, 2, figsize=(14,10))
        sns.histplot(self.df["alcohol"], kde=True, ax=axes[0,0])
        axes[0,0].set_title("Distribución Alcohol")

        sns.histplot(self.df["ph"], kde=True, ax=axes[0,1])
        axes[0,1].set_title("Distribución pH")

        sns.histplot(self.df["acidez_fija"], kde=True, ax=axes[1,0])
        axes[1,0].set_title("Distribución Acidez Fija")

        sns.histplot(self.df["acidez_volatil"], kde=True, ax=axes[1,1])
        axes[1,1].set_title("Distribución Acidez Volátil")

        plt.tight_layout()
        plt.savefig(RUTA_OUTPUTS / "eda_01_distribuciones.png")
        plt.close()

        # 2. Outliers
        fig, axes = plt.subplots(1,3, figsize=(15,5))
        sns.boxplot(x=self.df["alcohol"], ax=axes[0])
        axes[0].set_title("Outliers Alcohol")

        sns.boxplot(x=self.df["ph"], ax=axes[1])
        axes[1].set_title("Outliers pH")

        sns.boxplot(x=self.df["acidez_fija"], ax=axes[2])
        axes[2].set_title("Outliers Acidez Fija")

        plt.tight_layout()
        plt.savefig(RUTA_OUTPUTS / "eda_02_outliers.png")
        plt.close()

        # 3. Correlación
        plt.figure(figsize=(10,8))
        sns.heatmap(self.df.corr(numeric_only=True),
                    cmap="RdBu_r",
                    center=0)
        plt.title("Matriz de Correlación")
        plt.tight_layout()
        plt.savefig(RUTA_OUTPUTS / "eda_03_correlaciones.png")
        plt.close()

        # 4. Relaciones con calidad
        fig, axes = plt.subplots(1,2, figsize=(14,5))
        sns.regplot(data=self.df, x="alcohol", y="calidad",
                    scatter_kws={"alpha":0.4}, ax=axes[0])
        axes[0].set_title("Alcohol vs Calidad")

        sns.regplot(data=self.df, x="acidez_volatil", y="calidad",
                    scatter_kws={"alpha":0.4}, ax=axes[1])
        axes[1].set_title("Acidez Volátil vs Calidad")

        plt.tight_layout()
        plt.savefig(RUTA_OUTPUTS / "eda_04_relaciones_calidad.png")
        plt.close()

        print("✓ EDA completado y exportado.")
        return self

    # ─────────────────────────────
    # LIMPIEZA + EXPORT DATASET
    # ─────────────────────────────
    @registrar_ejecucion
    def limpiar_y_clasificar(self):

        print("\n═══ ETAPA DE LIMPIEZA DE DATOS ═══")

        # 1. Duplicados
        duplicados = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        print(f"→ Duplicados eliminados: {duplicados}")

        # 2. Nulos
        nulos = self.df.isna().sum().sum()
        self.df = self.df.dropna()
        print(f"→ Valores nulos eliminados: {nulos}")

        # 3. Outliers extremos (P98 alcohol)
        p98 = self.df["alcohol"].quantile(0.98)
        antes = len(self.df)
        self.df = self.df[self.df["alcohol"] <= p98]
        print(f"→ Outliers eliminados: {antes - len(self.df)}")

        # 4. Clasificación
        self.df["categoria"] = self.df.apply(
            lambda r: clasificar_vino(r.to_dict()), axis=1
        )

        # 5. Exportación dataset limpio
        self.df.to_csv(RUTA_DATA / "vinos_limpio.csv", index=False)
        print("✓ Dataset limpio exportado a data/vinos_limpio.csv")

        return self

    # ─────────────────────────────
    # SEMANA 3 - INTERPRETACIÓN IA
    # ─────────────────────────────
    @registrar_ejecucion
    def interpretar_con_ia(self):

        avg_alcohol = self.df["alcohol"].mean()
        avg_ph = self.df["ph"].mean()
        premium_count = len(self.df[self.df["categoria"] == "premium"])

        self.reporte_ia = {
            "diagnostico":
                f"Lote con alcohol promedio {avg_alcohol:.2f}% y pH promedio {avg_ph:.2f}.",
            "calidad_estimacion":
                "Alta" if premium_count > 20 else "Media",
            "recomendacion":
                "Optimizar fermentación para mejorar estabilidad química.",
            "cantidad_premium":
                premium_count
        }

        with open(RUTA_OUTPUTS / "reporte_ia.json", "w") as f:
            json.dump(self.reporte_ia, f, indent=4)

        print("✓ Reporte IA generado.")
        return self


# ─────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    pipeline = PipelineVinos("data/winequality-red.csv")

    (pipeline
        .ingestar()
        .eda()
        .limpiar_y_clasificar()
        .interpretar_con_ia()
    )

    print("\n✅ PROCESO COMPLETADO")