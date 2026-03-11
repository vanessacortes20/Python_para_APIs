"""
FUNCIONES PURAS DE LIMPIEZA
===========================
Funciones matemáticas puras (Semana 1). 
Características de estas funciones:
1. Son deterministas: Con las mismas entradas, producen la misma salida.
2. Sin efectos secundarios: No modifican el DataFrame global, siempre retornan uno nuevo.
"""
import pandas as pd

def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas que sean exactamente iguales (duplicados).
    Útil para no sobrerrepresentar una muestra.
    Retorna una copia nueva del DataFrame con el método .drop_duplicates().
    """
    return df.drop_duplicates()

def eliminar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina cualquier fila que contenga algún valor vacío/nulo (NaN).
    Garantiza la integridad antes de graficar con el método .dropna().
    """
    return df.dropna()

def eliminar_outliers_alcohol(df: pd.DataFrame, percentil: float = 0.98) -> pd.DataFrame:
    """
    Elimina los casos extremos (outliers) aislando el tope máximo superior.
    Calcula el cuantil especificado y filtra las filas de alcohol que lo superen.
    Por defecto corta por el percentil 98, estabilizando la varianza.
    """
    # Se calcula el punto de corte (límite máximo permitido)
    p_limite = df["alcohol"].quantile(percentil)
    # Se retornan solo las filas cuyo nivel de alcohol sea menor o igual al límite
    return df[df["alcohol"] <= p_limite]
