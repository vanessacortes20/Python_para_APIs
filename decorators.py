"""
MÓDULO DE DECORADORES (Semana 1)
================================
Contiene herramientas para:

• Logging de ejecución del pipeline
• Validación estadística automática (Test de Shapiro-Wilk)

Demuestra:
- Uso de decoradores simples
- Uso de Decorator Factory
- Aplicación de QA estadístico previo al análisis
"""

import functools
from datetime import datetime
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════
# 1. DECORADOR DE LOGGING
# ═══════════════════════════════════════════════════════════════

def registrar_ejecucion(func):
    """
    Decorador simple.
    
    Registra:
    - Hora de inicio
    - Nombre de la función
    - Tiempo total de ejecución
    
    Útil para trazabilidad del pipeline.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inicio = datetime.now()
        print(f"\n  [+] [{inicio:%H:%M:%S}] Ejecutando: {func.__name__}()")

        resultado = func(*args, **kwargs)

        duracion = (datetime.now() - inicio).total_seconds()
        print(f"  [✓] Finalizado en {duracion:.4f} segundos")

        return resultado

    return wrapper


# ═══════════════════════════════════════════════════════════════
# 2. DECORATOR FACTORY - VALIDACIÓN DE NORMALIDAD
# ═══════════════════════════════════════════════════════════════

def validar_normalidad(alpha: float = 0.05):
    """
    Decorator Factory.

    Realiza automáticamente el Test de Shapiro-Wilk
    sobre la variable 'ph' antes de ejecutar el método decorado.

    Parámetro:
    alpha → Nivel de significancia estadística (default 0.05)

    Interpretación:
    p < alpha  → Distribución NO normal
    p ≥ alpha  → Distribución compatible con normalidad
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            data = None

            # Verificar que exista instancia con DataFrame válido
            if args:
                instance = args[0]

                if hasattr(instance, "df"):
                    df = instance.df

                    if isinstance(df, pd.DataFrame) and not df.empty:
                        if "ph" in df.columns:
                            data = df["ph"].dropna()

            # Ejecutar test solo si hay datos suficientes
            if data is not None and len(data) >= 3:

                try:
                    # Shapiro es sensible a muestras muy grandes (>5000)
                    muestra = data.head(5000)
                    stat, p_value = stats.shapiro(muestra)

                    print("\n    ── Control Estadístico (Shapiro-Wilk) ──")
                    print(f"    Estadístico: {stat:.4f}")
                    print(f"    p-valor: {p_value:.4f}")

                    if p_value < alpha:
                        print(f"    ⚠ Distribución NO normal (p < {alpha})")
                        print("    → Se recomienda usar Mediana para análisis robusto.")
                    else:
                        print(f"    ✓ Distribución compatible con normalidad (p ≥ {alpha})")
                        print("    → Se puede usar Media como medida representativa.")

                except Exception as e:
                    print("    ⚠ Error al ejecutar test de normalidad:", e)

            else:
                print("    ⚠ No hay datos suficientes para prueba de normalidad.")

            return func(*args, **kwargs)

        return wrapper

    return decorator