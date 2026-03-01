"""
╔══════════════════════════════════════════════════════════════════════════╗
║   SISTEMA DE ANÁLISIS DE CALIDAD ENOLÓGICA (Vinos)                       ║
║   Análisis Químico y Clasificación de Calidad                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║   Curso  : Python para APIs e IA Aplicada                                ║
║   Semanas: 1, 2 y 3 (Entrega Integrada)                                  ║
║   Univ.  : Universidad Santo Tomás · 2026                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║   CONCEPTOS APLICADOS                                                    ║
║   - Semana 1: Pattern Matching (match/case) y Decoradores                ║
║   - Semana 2: OOP, Encadenamiento y Pydantic v2                          ║
║   - Semana 3: IA Generativa e Interpretación de Resultados               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ── 1. LIBRERÍAS ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
from scipy import stats

# ── 2. MÓDULOS PROPIOS ──────────────────────────────────────────────────────
from decorators import registrar_ejecucion, validar_normalidad
from schemas import VinoSchema  # Asegúrate de que el archivo se llame schemas.py

# ── 3. CONFIGURACIÓN ESTÉTICA ───────────────────────────────────────────────
RUTA_SALIDA = Path("outputs")
RUTA_SALIDA.mkdir(exist_ok=True)

PALETA = {
    "primario": "#641E16",  
    "secundario": "#A93226",
    "premium": "#D4AC0D",
    "estandar": "#2E86C1",
    "economico": "#85929E",
    "critico": "#CB4335",
    "neutro": "#64748B"
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC",
    "axes.edgecolor": "#CBD5E1",
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120
})

# ══════════════════════════════════════════════════════════════════════════
# 4. LÓGICA DE NEGOCIO (Pattern Matching - Semana 1)
# ══════════════════════════════════════════════════════════════════════════

def clasificar_vino(data: dict) -> str:
    """Clasifica el vino usando Pattern Matching."""
    match data:
        case {"calidad": c} if c >= 8:
            return "premium"
        case {"calidad": c} if c >= 6:
            return "estandar"
        case {"calidad": c} if c >= 4:
            return "economico"
        case {"calidad": None} | {}:
            return "sin_datos"
        case _:
            return "baja_calidad"

# ══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL (POO e IA - Semanas 2 y 3)
# ══════════════════════════════════════════════════════════════════════════

class PipelineVinos:
    """Orquestador del análisis de calidad de vinos con integración de IA."""

    def __init__(self, ruta_csv: str):
        self.ruta = ruta_csv
        self.df: pd.DataFrame = None
        self.df_crudo: pd.DataFrame = None
        self.registros_validados = []
        self.reporte_ia = {}  # Almacenará el resultado de la Semana 3

    # ── Etapa 1: Ingesta + Validación Pydantic ────────────────────────────
    @registrar_ejecucion
    def ingestar(self) -> "PipelineVinos":
        print("\n    [PYDANTIC] Validando registros químicos:")
        try:
            raw_df = pd.read_csv(self.ruta, sep=';')
            validos, errores = [], []
            for _, fila in raw_df.iterrows():
                try:
                    v = VinoSchema(**fila.to_dict())
                    validos.append(v.model_dump())
                except Exception:
                    errores.append(fila.to_dict())

            self.df_crudo = pd.DataFrame(validos)
            self.df = self.df_crudo.copy()
            print(f"    ✓ Carga exitosa: {len(validos)} válidos | {len(errores)} fallidos")
        except FileNotFoundError:
            print(f"    ✗ Archivo no encontrado. Iniciando modo demo sintético.")
            self._generar_datos_sinteticos()
        return self

    def _generar_datos_sinteticos(self):
        data = {
            "fixed_acidity": np.random.uniform(4, 15, 150),
            "volatile_acidity": np.random.uniform(0.1, 1.2, 150),
            "ph": np.random.normal(3.3, 0.2, 150),
            "alcohol": np.random.uniform(8, 14, 150),
            "calidad": np.random.randint(3, 9, 150)
        }
        self.df_crudo = pd.DataFrame(data)
        self.df = self.df_crudo.copy()

    # ── Etapa 2: EDA ─────────────────────────────────────────────────────
    @registrar_ejecucion
    @validar_normalidad(alpha=0.05)
    def eda(self) -> "PipelineVinos":
        print(f"    -> Análisis descriptivo (n={len(self.df)})")
        self._graficar_eda_completo()
        return self

    def _graficar_eda_completo(self):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(self.df["alcohol"], kde=True, color=PALETA["primario"], ax=ax1)
        ax1.set_title("Distribución de Alcohol")
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(x=self.df["ph"], color=PALETA["estandar"], ax=ax2)
        ax2.set_title("Outliers en pH")
        ax3 = fig.add_subplot(gs[1, :])
        sns.heatmap(self.df.corr(), annot=True, cmap="RdBu_r", ax=ax3)
        plt.tight_layout()
        fig.savefig(RUTA_SALIDA / "eda_vinos_completo.png")
        plt.close()

    # ── Etapa 3: Limpieza y Clasificación ───────────────────────────────
    @registrar_ejecucion
    def limpiar_y_clasificar(self) -> "PipelineVinos":
        self.df = self.df.drop_duplicates().dropna()
        self.df['categoria'] = self.df.apply(lambda r: clasificar_vino(r.to_dict()), axis=1)
        return self

    # ── Etapa 4: INTERPRETACIÓN IA (SEMANA 3) ───────────────────────────
    @registrar_ejecucion
    def interpretar_con_ia(self) -> "PipelineVinos":
        """Genera un diagnóstico experto basado en los datos procesados."""
        print("\n    [IA] Generando interpretación experta del lote...")
        
        # 1. Preparar métricas clave para el prompt
        avg_alcohol = self.df["alcohol"].mean()
        avg_ph = self.df["ph"].mean()
        premium_count = len(self.df[self.df["categoria"] == "premium"])
        
        # 2. Simulación de llamada a LLM (Semana 3)
        # Aquí se integraría OpenAI/LangChain. Usamos un motor de reglas experto:
        self.reporte_ia = {
            "diagnostico": f"Lote con alcohol promedio del {avg_alcohol:.1f}% y estabilidad de pH en {avg_ph:.2f}.",
            "calidad_ia": "Alta" if premium_count > 10 else "Media-Estandar",
            "sugerencia": "Se recomienda estabilización tartárica en frío debido a los niveles de pH observados.",
            "market_insight": f"Se han detectado {premium_count} muestras con potencial de Gran Reserva."
        }
        
        print(f"    ✓ IA completada: {self.reporte_ia['calidad_ia']}")
        return self

    # ── Etapa 5: Visualización y Exportación ───────────────────────────
    @registrar_ejecucion
    def visualizar_y_exportar(self) -> "PipelineVinos":
        # Gráfica de resultados final
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.countplot(data=self.df, x='categoria', palette="viridis", ax=axes[0])
        axes[0].set_title("Clasificación Final (Pattern Matching)")
        sns.regplot(data=self.df, x='alcohol', y='calidad', ax=axes[1], color=PALETA["primario"])
        axes[1].set_title("Correlación Alcohol vs Calidad")
        plt.savefig(RUTA_SALIDA / "panel_resultados_final.png")
        plt.close()

        # Exportación
        with open(RUTA_SALIDA / "reporte_ia.json", "w") as f:
            json.dump(self.reporte_ia, f, indent=4)
        with open(RUTA_SALIDA / "modelo_final.pkl", "wb") as f:
            pickle.dump(self.df, f)
        
        return self

    def imprimir_reporte_ejecutivo(self):
        """Muestra el reporte final de la Semana 3 por consola."""
        print("\n" + "═"*65)
        print(" 🤖 REPORTE DE INTELIGENCIA ARTIFICIAL (SEMANA 3)")
        print("═"*65)
        print(f" ESTATUS    : {self.reporte_ia['calidad_ia'].upper()}")
        print(f" DIAGNÓSTICO: {self.reporte_ia['diagnostico']}")
        print(f" SUGERENCIA : {self.reporte_ia['sugerencia']}")
        print(f" MERCADO    : {self.reporte_ia['market_insight']}")
        print("═"*65 + "\n")

# ══════════════════════════════════════════════════════════════════════════
# 6. PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*60)
    print(" SISTEMA INTEGRADO DE CALIDAD ENOLÓGICA (V1.0)")
    print("█"*60)

    pipeline = PipelineVinos("data/winequality-red.csv")
    
    try:
        # FLUJO ENCADENADO COMPLETO (Semanas 1, 2 y 3)
        (pipeline
            .ingestar()
            .eda()
            .limpiar_y_clasificar()
            .interpretar_con_ia()      # <--- Semana 3
            .visualizar_y_exportar())  # <--- Salida final

        # Resultado narrativo de la IA
        pipeline.imprimir_reporte_ejecutivo()

        # Demo Pattern Matching Manual (Semana 1)
        print("📋  Validación Manual:")
        test = {"calidad": 8, "alcohol": 13.5}
        print(f"  Entrada: {test} -> Categoría: {clasificar_vino(test).upper()}")

        print(f"\n✅ PROCESO FINALIZADO. Archivos generados en: {RUTA_SALIDA.resolve()}")

    except Exception as e:
        print(f"\n✗ Error crítico: {e}")