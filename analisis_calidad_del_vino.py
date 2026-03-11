"""
╔══════════════════════════════════════════════════════════════╗
║ SISTEMA INTEGRADO DE ANÁLISIS ENOLÓGICO                     ║
║ Semanas 1, 2 y 3                                             ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
"""
PIPELINE PRINCIPAL DE ANÁLISIS ENOLÓGICO
========================================
Este script actúa como el "Orquestador" principal del proyecto.
Carga los datos, los valida (Pydantic), hace el EDA, los limpia de duplicados 
y valores nulos usando las funciones importadas, y finalmente ejecuta la IA.
"""
# 1. LIBRERÍAS
# ─────────────────────────────────────────────
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Importamos nuestras herramientas creadas en módulos separados
from decorators import registrar_ejecucion, validar_normalidad
from schemas import VinoInputSchema, VinoOutputSchema
from limpieza import eliminar_duplicados, eliminar_nulos, eliminar_outliers_alcohol


# ─────────────────────────────────────────────
# 2. CONFIGURACIÓN DE RUTAS LOCALES
# ─────────────────────────────────────────────
# Pathlib nos permite usar rutas relativas sin importar en qué SO estemos.
RUTA_BASE = Path(__file__).resolve().parent
RUTA_DATA = RUTA_BASE / "data"
RUTA_OUTPUTS = RUTA_BASE / "outputs"
RUTA_OUTPUTS.mkdir(exist_ok=True)

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
# ═══════════════════════════════════════════════════════════════
# 1. CLASE PRINCIPAL DEL PIPELINE (POO)
# ═══════════════════════════════════════════════════════════════

class PipelineVinos:
    """
    Clase que agrupa todas las fases de procesamiento secuencial.
    Sigue el patrón 'Builder' devolviendo `self` para permitir 
    concatenar métodos de manera fluida (method chaining).
    """

    def __init__(self, ruta_csv: str):
        self.ruta_csv = RUTA_DATA / ruta_csv
        self.df = None
        
        # Creación de carpetas de salida en caso de no existir
        RUTA_OUTPUTS.mkdir(exist_ok=True)
        self.reporte_eda = {}
        self.reporte_ia = {}

    # ─────────────────────────────
    # INGESTA + VALIDACIÓN (Semana 2)
    # ─────────────────────────────
    @registrar_ejecucion
    def ingestar_y_validar(self):
        """
        Fase 1 e Integración Semana 2 (Pydantic).
        Carga el CSV crudo y valida INMEDIATAMENTE de manera estricta
        cada fila para garantizar que cumpla con VinoInputSchema.
        """
        try:
            print("\n  [INFO] Cargando dataset crudo...")
            # Carga rápida a Pandas DataFrame
            df_crudo = pd.read_csv(self.ruta_csv)
            
            # List Comprehension para intentar parsear fila a fila a dict()
            # Y descartar de inmediato filas dañadas
            filas_validas = []
            for idx, fila in df_crudo.iterrows():
                try:
                    # Empaquetamos en el Schema. Si rompe reglas químicas, tira error.
                    vino_validado = VinoInputSchema(**fila.to_dict())
                    filas_validas.append(vino_validado.model_dump())
                except Exception as e:
                    # Fallos en QA simplemente no entran a la base
                    pass
            
            # Recreamos el DataFrame solo con registros de alta pureza
            self.df = pd.DataFrame(filas_validas)
            print(f"    ✓ Registros validados según esquema Pydantic: {len(self.df)}")
            return self

        except FileNotFoundError:
            print(f"  [ERROR] No se encontró el dataset en: {self.ruta_csv}")
            sys.exit(1)

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
    def aplicar_limpieza(self):
        """
        Fase 3: Limpieza (Semana 1).
        Llama a las funciones matemáticas sin estado guardadas en limpieza.py
        """
        if self.df is None: return self

        # 1. Remover Nulls
        antes = len(self.df)
        self.df = eliminar_nulos(self.df)
        print(f"    ✓ Nulos eliminados: {antes - len(self.df)}")

        # 2. Remover Duplicados
        antes = len(self.df)
        self.df = eliminar_duplicados(self.df)
        print(f"    ✓ Duplicados eliminados: {antes - len(self.df)}")

        # 3. Remover Outliers en Alcohol
        antes = len(self.df)
        self.df = eliminar_outliers_alcohol(self.df)
        print(f"    ✓ Outliers de alcohol eliminados: {antes - len(self.df)}")

        # 4. Guardar archivo final inmaculado
        ruta_salida = RUTA_DATA / "vinos_limpio.csv"
        self.df.to_csv(ruta_salida, index=False)
        print(f"    ✓ Dataset final guardado localmente exportado en carpeta /data.")
        return self

    # ─────────────────────────────
    # SEMANA 3 - INTERPRETACIÓN IA
    # ─────────────────────────────
    @registrar_ejecucion
    def interpretacion_ai(self):
        """
        Fase 4: Semántica y Salida Final.
        Usa Pattern Matching (Estructura match-case nativa de Python 3.10+) 
        y comprension de listas para aplicar reglas de negocio súper rápido.
        Además evalúa su salida en base al VinoOutputSchema.
        """
        if self.df is None: return self

        # Extraemos valores como variables nativas de python con zip para iteración rápida
        calidades = self.df['calidad'].values
        alcoholes = self.df['alcohol'].values

        categorias = []
        for q, alc in zip(calidades, alcoholes):
            # Uso avanzado de Match Case evaluando Tuplas (Calidad, Alcohol)
            match (q, alc):
                case (q, alc) if q >= 7 and alc >= 11.5:
                    categorias.append("PREMIUM RESERVA")
                case (q, _) if q >= 6:
                    categorias.append("SELECCIÓN")
                case (q, alc) if q <= 4:
                    categorias.append("DESCARTE")
                case _:
                    categorias.append("MESA (ESTÁNDAR)")
        
        # Asignamos la nueva variable al DataFrame final
        self.df['categoria'] = categorias

        # Validación final de salida contra OutputSchema de Pydantic
        muestra = [VinoOutputSchema(**fila.to_dict()).model_dump() for _, fila in self.df.head(5).iterrows()]

        # Genera el JSON final con métricas descriptivas
        reporte = {
            "tamaño_final": len(self.df),
            "promedios": self.df.mean(numeric_only=True).to_dict(),
            "distribucion_categorias": self.df['categoria'].value_counts().to_dict(),
            "muestra_valida_pydantic": muestra
        }

        with open(RUTA_OUTPUTS / "reporte_ia.json", "w", encoding="utf-8") as f:
            json.dump(reporte, f, indent=4)
        
        print("    ✓ Machine Pattern Matching completado.")
        print("    ✓ Reporte IA generado en formato JSON validado.")
        return self


# ─────────────────────────────────────────────
# EJECUCIÓN DEL PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Se instancia la clase con el set de datos inicial
    pipeline = PipelineVinos("dataset_calidad_vinos.csv")

    # Gracias al patrón Builder, podemos encadenar los métodos de forma muy limpia.
    # El orden lógico de la ETL: Ingresa -> Analiza -> Limpia -> Interpreta
    (pipeline
        .ingestar_y_validar()
        .eda()
        .aplicar_limpieza()
        .interpretacion_ai()
    )

    print("\n✅ PROCESO COMPLETADO Y DOCUMENTADO")