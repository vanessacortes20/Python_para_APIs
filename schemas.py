"""
MÓDULO DE MODELOS Y VALIDACIÓN (QA) - Semana 2
==============================================
Define las estructuras de datos estrictas para el análisis enológico.
Utiliza Pydantic v2 para garantizar que cada fila proveniente de un CSV
o una petición API cumpla con los tipos de datos exactos (int, float) 
y límites de dominio lógicos de la química del vino.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# SCHEMA PRINCIPAL DE VALIDACIÓN
# ═══════════════════════════════════════════════════════════════

class VinoInputSchema(BaseModel):
    """
    Contrato de datos de entrada alineado con dataset en español.
    Si una fila del CSV intenta pasar por este filtro rompiendo las 
    reglas descritas abajo (ej. pH de 10), Pydantic arrojará un error.
    """

    # Field(ge=0) significa (greater than or equal a 0). Evita números negativos.
    acidez_fija: float = Field(ge=0)
    acidez_volatil: float = Field(ge=0)
    
    # Optional[] dicta que esta columna puede venir vacía sin romper el programa,
    # asumiendo por defecto (default=0.0) el valor.
    acido_citrico: Optional[float] = Field(default=0.0, ge=0)
    
    azucar_residual: float = Field(ge=0)
    cloruros: float = Field(ge=0)

    dioxido_azufre_libre: float = Field(ge=0)
    dioxido_azufre_total: float = Field(ge=0)
    densidad: float = Field(ge=0)

    # Validaciones biológicas fuertes (limits lógicos)
    ph: float = Field(ge=2.5, le=4.5)           # No existen vinos con pH de 9.
    sulfatos: float = Field(ge=0)
    alcohol: float = Field(ge=7.0, le=20.0)     # Excluye agua común o alcohol puro industrial.

    calidad: int = Field(ge=0, le=10)           # La calificación de expertos solo va de 0 a 10.

    model_config = ConfigDict(
        # Limpia espacios vacíos accidentales al principio o final del texto
        str_strip_whitespace=True
    )


class VinoOutputSchema(VinoInputSchema):
    """
    Contrato de datos de salida que hereda todas las valiaciones
    del InputSchema anterior y le añade la nueva categorización 
    generada por Pattern Matching.
    """
    categoria: str


class RespuestaLimpiezaSchema(BaseModel):
    """
    Esquema principal de respuesta para la API de FastAPI (/clean).
    Este modelo es el que se pinta automáticamente de verde en Swagger UI (/docs)
    explicándole al cliente web qué campos JSON va a recibir exactamente en pantalla.
    """
    status: str
    mensaje: str
    duplicados_eliminados: int
    nulos_eliminados: int
    filas_finales: int
    archivo_guardado: str