"""
MÓDULO DE MODELOS Y VALIDACIÓN (QA)
===================================
Define las estructuras de datos y reglas de validación para el 
análisis enológico utilizando Pydantic v2.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════
# 1. SCHEMA DE VALIDACIÓN (Pydantic)
# ══════════════════════════════════════════════════════════════════════════

class VinoSchema(BaseModel):
    """
    Define el contrato de datos para un registro de vino.
    Realiza validaciones de tipo y rango en tiempo real.
    """
    # Usamos Field con alias para que coincida con los nombres comunes en CSV
    acidez_fija: float = Field(alias="fixed acidity", ge=0)
    acidez_volatil: float = Field(alias="volatile acidity", ge=0)
    acido_citrico: Optional[float] = Field(alias="citric acid", default=0.0)
    azucar_residual: float = Field(alias="residual sugar", ge=0)
    cloruros: float = Field(default=0.0)
    ph: float = Field(ge=2.5, le=4.5) # Rango químico real del vino
    alcohol: float = Field(ge=7.0, le=20.0) # Rango de fermentación
    calidad: int = Field(ge=0, le=10)

    # Configuración para permitir el uso de alias al cargar datos
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )

    @field_validator("ph")
    @classmethod
    def validar_ph(cls, v: float) -> float:
        """Validación extra personalizada para el pH."""
        if v < 2.8:
            # Podríamos lanzar un aviso o ajustar el dato
            pass
        return round(v, 2)

# ══════════════════════════════════════════════════════════════════════════
# 2. MODELO DE RESULTADOS (Para Exportación JSON)
# ══════════════════════════════════════════════════════════════════════════

class ResultadoAnalisis(BaseModel):
    """Estructura para el reporte final de exportación."""
    fecha_proceso: datetime = Field(default_factory=datetime.now)
    total_muestras: int
    promedio_alcohol: float
    distribucion_categorias: dict
    archivo_grafica: str

# ══════════════════════════════════════════════════════════════════════════
# 3. FUNCIONES DE APOYO (QA Lote)
# ══════════════════════════════════════════════════════════════════════════

def validar_lote_vinos(lista_datos: List[dict]):
    """
    Valida una lista de diccionarios y separa los correctos de los erróneos.
    """
    validos = []
    errores = []
    
    for item in lista_datos:
        try:
            vino = VinoSchema(**item)
            validos.append(vino)
        except Exception as e:
            errores.append({"dato": item, "error": str(e)})
            
    return validos, errores