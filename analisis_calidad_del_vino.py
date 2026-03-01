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
║   - Pattern Matching (match/case con guardas)   → clasificar_vino()      ║
║   - Decoradores (Simple y Factory)              → decorators.py          ║
║   - OOP (Pipeline con encadenamiento)           → PipelineVinos          ║
║   - QA Automático con Pydantic v2               → modelos.py             ║
║   - EDA avanzado con subplots y mapas de calor  → eda() / visualizar()   ║
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
from schemas import VinoSchema  # Basado en tu estructura de Pydantic

# ── 3. CONFIGURACIÓN ESTÉTICA ───────────────────────────────────────────────
RUTA_SALIDA = Path("outputs")
RUTA_SALIDA.mkdir(exist_ok=True)

PALETA = {
    "primario": "#641E16",  # Color vino tinto
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
    """Clasifica el vino usando Pattern Matching (Semana 1)."""
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
# 5. PIPELINE PRINCIPAL (POO - Semana 2)
# ══════════════════════════════════════════════════════════════════════════

class PipelineVinos:
    """Orquestador del análisis de calidad de vinos mediante POO."""

    def __init__(self, ruta_csv: str):
        self.ruta = ruta_csv
        self.df: pd.DataFrame = None
        self.df_crudo: pd.DataFrame = None
        self.registros_validados = []

    # ── Etapa 1: Ingesta + Validación Pydantic ────────────────────────────
    @registrar_ejecucion
    def ingestar(self) -> "PipelineVinos":
        print("\n   [PYDANTIC] Validando registros químicos:")
        try:
            # Leemos con separador ; común en datasets de vinos
            raw_df = pd.read_csv(self.ruta, sep=';')
            
            validos, errores = [], []
            for i, fila in raw_df.iterrows():
                try:
                    # Validación vía Pydantic (Semana 2)
                    v = VinoSchema(**fila.to_dict())
                    validos.append(v.model_dump())
                except Exception:
                    errores.append(fila.to_dict())

            self.registros_validados = validos
            self.df_crudo = pd.DataFrame(validos)
            self.df = self.df_crudo.copy()
            
            print(f"    ✓ Carga exitosa: {len(validos)} válidos | {len(errores)} fallidos")
        except FileNotFoundError:
            print(f"    ✗ Error: No se encontró el archivo en {self.ruta}")
            # Crear datos sintéticos de emergencia para no romper el flujo
            self._generar_datos_sinteticos()
        
        return self

    def _generar_datos_sinteticos(self):
        print("    ⚠ Generando datos sintéticos para demostración...")
        data = {
            "fixed_acidity": np.random.uniform(4, 15, 100),
            "volatile_acidity": np.random.uniform(0.1, 1.2, 100),
            "ph": np.random.normal(3.3, 0.2, 100),
            "alcohol": np.random.uniform(8, 14, 100),
            "calidad": np.random.randint(3, 9, 100)
        }
        self.df_crudo = pd.DataFrame(data)
        self.df = self.df_crudo.copy()

    # ── Etapa 2: EDA — Análisis Exploratorio ─────────────────────────────
    @registrar_ejecucion
    @validar_normalidad(alpha=0.05)
    def eda(self) -> "PipelineVinos":
        """Análisis estadístico descriptivo y test de normalidad."""
        print("\n" + "─" * 56)
        print("  EDA — HALLAZGOS QUÍMICOS")
        print("─" * 56)
        
        # H1: Dimensiones
        print(f"  [H1] Dataset: {self.df.shape[0]} registros procesados")
        
        # H2: Análisis de Alcohol
        avg_alc = self.df["alcohol"].mean()
        print(f"  [H2] Grado Alcohólico Promedio: {avg_alc:.2f}%")

        # El decorador @validar_normalidad actuará sobre la columna 'ph' 
        # automáticamente si está configurado para leer self.df
        self._graficar_eda_completo()
        return self

    def _graficar_eda_completo(self):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Distribución de Alcohol (Hist + KDE)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(self.df["alcohol"], kde=True, color=PALETA["primario"], ax=ax1)
        ax1.set_title("Distribución de Grado Alcohólico")

        # 2. Boxplot de pH para Outliers
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(x=self.df["ph"], color=PALETA["estandar"], ax=ax2)
        ax2.set_title("Detección de Outliers en pH")

        # 3. Mapa de Calor (Correlaciones químicas)
        ax3 = fig.add_subplot(gs[1, :])
        sns.heatmap(self.df.corr(), annot=True, cmap="RdBu_r", fmt=".2f", ax=ax3)
        ax3.set_title("Matriz de Correlación de Atributos Químicos")

        plt.tight_layout()
        fig.savefig(RUTA_SALIDA / "eda_vinos_completo.png")
        plt.close()
        print(f"    ✓ Gráficas EDA guardadas en {RUTA_SALIDA}")

    # ── Etapa 3: Limpieza y Clasificación ───────────────────────────────
    @registrar_ejecucion
    def limpiar_y_clasificar(self) -> "PipelineVinos":
        """Limpieza de nulos y aplicación de lógica de negocio."""
        conteo_pre = len(self.df)
        self.df = self.df.drop_duplicates().dropna()
        
        # Aplicamos Pattern Matching registro a registro
        self.df['categoria'] = self.df.apply(lambda r: clasificar_vino(r.to_dict()), axis=1)
        
        print(f"    ✓ Limpieza: {conteo_pre - len(self.df)} duplicados eliminados")
        print(f"    ✓ Clasificación: {self.df['categoria'].nunique()} categorías asignadas")
        return self

    # ── Etapa 4: Visualización de Resultados Finales ────────────────────
    @registrar_ejecucion
    def visualizar_resultados(self) -> "PipelineVinos":
        """Genera el panel comparativo final."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribución de categorías
        sns.countplot(data=self.df, x='categoria', 
                      palette=[PALETA["premium"], PALETA["estandar"], PALETA["economico"]], 
                      ax=axes[0], order=["premium", "estandar", "economico", "baja_calidad"])
        axes[0].set_title("Conteo por Categoría de Mercado")

        # Calidad vs Alcohol
        sns.regplot(data=self.df, x='alcohol', y='calidad', 
                    scatter_kws={'alpha':0.3, 'color': PALETA["primario"]}, 
                    line_kws={'color': 'black'}, ax=axes[1])
        axes[1].set_title("Relación Alcohol vs Calidad Percibida")

        plt.tight_layout()
        fig.savefig(RUTA_SALIDA / "panel_resultados_final.png")
        plt.close()
        return self

    # ── Etapa 5: Exportación ──────────────────────────────────────────
    @registrar_ejecucion
    def exportar(self):
        """Serialización de datos en múltiples formatos."""
        # 1. Resumen Estadístico en JSON
        resumen = self.df.describe().to_dict()
        with open(RUTA_SALIDA / "reporte_estadistico.json", "w") as f:
            json.dump(resumen, f, indent=4)
        
        # 2. Dataset final en Pickle para persistencia
        with open(RUTA_SALIDA / "backup_analisis.pkl", "wb") as f:
            pickle.dump(self.df, f)
            
        print(f"    ✓ Reportes generados en: {RUTA_SALIDA.resolve()}")
        return self

# ══════════════════════════════════════════════════════════════════════════
# 6. EJECUCIÓN (Main)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*60)
    print(" INICIANDO PIPELINE DE CALIDAD ENOLÓGICA")
    print("═"*60)

    # Instanciamos el pipeline con la ruta al dataset
    pipeline = PipelineVinos("data/winequality-red.csv")
    
    # FLUJO ENCADENADO (Method Chaining - Semana 2)
    try:
        (pipeline
            .ingestar()
            .eda()
            .limpiar_y_clasificar()
            .visualizar_resultados()
            .exportar())

        # ── Demo de Pattern Matching manual (Semana 1) ──────────────────
        print("\n📋  DEMO: Test de Clasificación Individual")
        print("─" * 40)
        muestras_test = [
            {"calidad": 9, "alcohol": 14.2},
            {"calidad": 5, "alcohol": 10.0},
            {"calidad": 2, "alcohol": 9.5},
            {}
        ]
        
        for m in muestras_test:
            res = clasificar_vino(m)
            simbolo = "⭐" if res == "premium" else "🍷"
            print(f"  {simbolo} Entrada: {str(m):28} -> Categoría: {res.upper()}")

        print(f"\n✅ PROCESO FINALIZADO EXITOSAMENTE.")

    except Exception as e:
        print(f"\n✗ Error crítico en el pipeline: {e}")