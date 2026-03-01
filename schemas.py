"""
MÓDULO DE MODELOS Y VALIDACIÓN (QA)
===================================
Define las estructuras de datos y reglas de validación
para el análisis enológico utilizando Pydantic v2.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# SCHEMA DE VALIDACIÓN
# ═══════════════════════════════════════════════════════════════

class VinoSchema(BaseModel):
    """
    Contrato de datos alineado con dataset en español.
    """

    acidez_fija: float = Field(ge=0)
    acidez_volatil: float = Field(ge=0)
    acido_citrico: Optional[float] = Field(default=0.0, ge=0)
    azucar_residual: float = Field(ge=0)
    cloruros: float = Field(ge=0)

    dioxido_azufre_libre: float = Field(ge=0)
    dioxido_azufre_total: float = Field(ge=0)
    densidad: float = Field(ge=0)

    ph: float = Field(ge=2.5, le=4.5)
    sulfatos: float = Field(ge=0)
    alcohol: float = Field(ge=7.0, le=20.0)

    calidad: int = Field(ge=0, le=10)

    model_config = ConfigDict(
        str_strip_whitespace=True
    )