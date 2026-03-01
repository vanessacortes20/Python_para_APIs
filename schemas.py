"""
MÓDULO DE MODELOS Y VALIDACIÓN (QA)
===================================

Define:
• Contrato de datos con Pydantic v2
• Validaciones químicas realistas
• Modelo estructurado para exportación
• Validación por lote

Demuestra:
- Uso avanzado de Pydantic
- Validaciones personalizadas
- Tipado fuerte
- Diseño orientado a QA
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Tuple
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# 1. SCHEMA DE VALIDACIÓN
# ═══════════════════════════════════════════════════════════════

class VinoSchema(BaseModel):
    """
    Representa un registro individual del dataset de vinos.
    
    Incluye:
    - Validación de tipos
    - Validación de rangos químicos
    - Alias para coincidir con el CSV original
    """

    # Alias coinciden EXACTAMENTE con winequality-red.csv
    acidez_fija: float = Field(alias="fixed acidity", ge=0)
    acidez_volatil: float = Field(alias="volatile acidity", ge=0)
    acido_citrico: float = Field(alias="citric acid", ge=0)
    azucar_residual: float = Field(alias="residual sugar", ge=0)
    cloruros: float = Field(alias="chlorides", ge=0)
    ph: float = Field(alias="pH", ge=2.5, le=4.5)
    alcohol: float = Field(ge=7.0, le=20.0)
    calidad: int = Field(alias="quality", ge=0, le=10)

    # Configuración Pydantic v2
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )

    # ─────────────────────────────
    # Validaciones adicionales
    # ─────────────────────────────
    @field_validator("ph")
    @classmethod
    def validar_ph(cls, v: float) -> float:
        """
        Ajusta precisión y valida rango químico coherente.
        """
        return round(v, 2)

    @field_validator("alcohol")
    @classmethod
    def validar_alcohol(cls, v: float) -> float:
        """
        Redondeo controlado para consistencia analítica.
        """
        return round(v, 2)


# ═══════════════════════════════════════════════════════════════
# 2. MODELO DE RESULTADOS ESTRUCTURADO
# ═══════════════════════════════════════════════════════════════

class ResultadoAnalisis(BaseModel):
    """
    Modelo estructurado para exportación de resultados finales.
    """
    fecha_proceso: datetime = Field(default_factory=datetime.now)
    total_muestras: int
    promedio_alcohol: float
    promedio_ph: float
    distribucion_categorias: dict
    conclusiones_ia: dict


# ═══════════════════════════════════════════════════════════════
# 3. VALIDACIÓN POR LOTE
# ═══════════════════════════════════════════════════════════════

def validar_lote_vinos(lista_datos: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Valida múltiples registros y separa:

    Returns:
        validos → Lista de diccionarios ya normalizados
        errores → Lista con detalle de errores encontrados
    """

    validos = []
    errores = []

    for item in lista_datos:
        try:
            vino = VinoSchema(**item)
            validos.append(vino.model_dump())
        except Exception as e:
            errores.append({
                "dato": item,
                "error": str(e)
            })

    return validos, errores