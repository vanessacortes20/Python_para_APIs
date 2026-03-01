"""
MÓDULO DE DECORADORES (Semana 1)
================================
Contiene herramientas para logging de ejecución y pruebas 
estadísticas automáticas (Shapiro-Wilk).
"""

import functools
from datetime import datetime
import pandas as pd
from scipy import stats

def registrar_ejecucion(func):
    """
    Decorador simple: Registra el inicio, fin y duración 
    de cada etapa del pipeline de vinos.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ts = datetime.now()
        print(f"\n  [+-> [{ts:%H:%M:%S}] Ejecutando: {func.__name__}()")
        
        resultado = func(*args, **kwargs)
        
        duracion = (datetime.now() - ts).total_seconds()
        print(f"  [+-] Etapa completada en {duracion:.4f}s")
        return resultado
    return wrapper

def validar_normalidad(alpha: float = 0.05):
    """
    Decorator Factory: Realiza el test de Shapiro-Wilk antes de 
    ejecutar funciones de análisis.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Intentamos extraer los datos del dataframe del objeto (self)
            # asumiendo que el primer argumento de la función decorada es 'self'
            data = None
            if len(args) > 0:
                instance = args[0]
                # Buscamos si la instancia tiene un atributo 'df' con la columna 'ph'
                if hasattr(instance, "df") and isinstance(instance.df, pd.DataFrame):
                    if "ph" in instance.df.columns:
                        data = instance.df["ph"].dropna()

            if data is not None and len(data) >= 3:
                # Tomamos una muestra para el test (Shapiro es sensible a n > 5000)
                muestra = data.head(5000)
                stat, p = stats.shapiro(muestra)
                
                print(f"    [QA STATS] Test de Normalidad (Variable: pH):")
                if p < alpha:
                    print(f"      ⚠ pH NO normal (p={p:.4f} < α={alpha}). Se sugiere usar Mediana.")
                else:
                    print(f"      ✓ pH Normal (p={p:.4f} ≥ α={alpha}). Se usará Media.")
            else:
                print("    ⚠ No se encontraron datos suficientes para test de normalidad.")

            return func(*args, **kwargs)
        return wrapper
    return decorator