# 🍷 Sistema Integrado de Análisis Enológico (Calidad del Vino)
### Curso Python para APIs e IA Aplicada — Universidad Santo Tomás (USTA)

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-Finalizado-%2328a745?style=for-the-badge)

---

## 🎯 ¿Qué hace este proyecto?

Transforma un dataset crudo sobre la calidad del vino tinto (`dataset_calidad_vinos.csv`) en un **pipeline reproducible de análisis robusto**, aplicando todos los conceptos del curso (Semanas 1 a la 4).

El motor ingesta los datos, los valida rigurosamente descartando anomalías físicas/biológicas (Semana 2), genera reportes exploratorios (EDA), limpia los datos basándose en evidencias matemáticas puras (Semana 1) y finalmente, **expone esta lógica a través de una API RESTful moderna, asíncrona y autrodocumentada** (Semanas 3 y 4).

---

## 🗂 Estructura del Proyecto

El código está modularizado para respetar el Principio de Responsabilidad Única, resultando en un sistema elegante y fácil de testear:

```text
/
│ 
├── 🚀 fastapi_app.py                ← Servidor Asíncrono autocomentado (Semana 4)
├── ⚙️ analisis_calidad_del_vino.py  ← Orquestador Principal (POO y Pattern Matching)
├── 🛡️ schemas.py                    ← Tipado estricto y reglas de negocio con Pydantic (Semana 2)
├── 🧼 limpieza.py                   ← Matemáticas Puras: Funciones sin 'side-effects'
├── 🎨 decorators.py                 ← Inyección de QA Estadístico y Logging (Semana 1)
│
├── data/                            ← Datasets CSV de origen y los resultantes procesados
│
└── outputs/                         ← Artefactos y gráficas generadas (Reporte EDA e IA)
    ├── reporte_ia.json
    ├── eda_01_distribuciones.png
    └── [..] otras graficas exploratorias
```

---

## 🚀 Instalación y Ejecución Local

Para probar la infraestructura localmente, sigue estos pasos explicativos:

```bash
# 1. Clona el repositorio a tu máquina
git clone https://github.com/vanessacortes20/Python_para_APIs.git
cd Python_para_APIs

# 2. Activa tu entorno virtual e instala las dependencias
pip install -r requirements.txt

# =========================================================================
# 🟢 OPCIÓN A: Ejecutar el pipeline de Ingeniería de Datos (Backend Local)
# =========================================================================
# Generará las gráficas, validará todo el dataset y creará los reportes JSON.
python analisis_calidad_del_vino.py

# =========================================================================
# 🔵 OPCIÓN B: Levantar el servidor web y acceder a la API interactiva
# =========================================================================
# Iniciará Uvicorn en el puerto 8000.
python -m uvicorn fastapi_app:app --reload

# 👉 VE A TU NAVEGADOR: http://127.0.0.1:8000/docs
# Allí verás la interfaz de Swagger lista para probar el endpoint /clean
```

---

## 🧠 Conceptos Evaluados y Aplicados

Este proyecto integra **exactamente** la rúbrica de evaluación provista durante las 4 semanas:

### 🌟 Semana 1: Fundamentos, Decoradores y Modularidad
- **Modularización:** Toda la lógica de Pandas se encapsuló en funciones matemáticas puras dentro de `limpieza.py` (cero variables globales, no mutan el estado original).
- **Decoradores (Metadata):** Se usó `@functools.wraps` en `decorators.py` para inyectar trazabilidad (logging) y un **Decorator Factory** para correr automáticamente pruebas de *Shapiro-Wilk*.

### 🌟 Semana 2: Tipo de Datos Fuertes e Integridad
- **Pydantic v2:** Se diseñó el módulo `schemas.py` (`VinoInputSchema`). No se acepta un solo registro donde el Vino tenga un `pH` de batería (mayor a 4.5) o un `alcohol` de garrafa industrial (>20%). 
- Los datos ilógicos son atrapados en la ingesta **antes** de llegar al sistema de limpiezas de Pandas.

### 🌟 Semana 3: Introducción a Web y APIs
- Uso del entorno web local.
- Separación de responsabilidad entre la lógica analítica de escritorio (`analisis_calidad_del_vino`) y el modelo Cliente/Servidor.

### 🌟 Semana 4: FastAPI, Asincronía y Pattern Matching
1. **Migración a FastAPI:** Se reemplazó el antiguo servidor síncrono por la velocidad de `fastapi` y `uvicorn`. Endpoint principal: `@app.post("/clean")`.
2. **Swagger UI Autogenerado:** El cliente puede probar el producto final simplemente visitando `/docs`.
3. **Response Models:** Se mapeó el retorno exhaustivo mediante `RespuestaLimpiezaSchema`.
4. **Pattern Matching (`match/case`):** Se utilizó esta potente característica de *Python 3.10+* durante la fase 4 (`interpretacion_ai()`) evaluando **Tuplas** `(Calidad, Alcohol)` en microsegundos para clasificar cada botella como "PREMIUM", "SELECCIÓN" o "DESCARTE".

---

## 📈 Hallazgos Destacados del EDA

El sistema exploratorio graficó y detectó lo siguiente (`outputs/`):

1. **Valores Atípicos (Outliers):** La columna `alcohol` presentó picos irreales. El sistema limpió matemáticamente estos errores aislando todo lo que superara el percentil 98 (P98).
2. **Correlaciones de Oro:** La matriz (`eda_03_correlaciones.png`) demostró que entre más nivel de *Alcohol*, mejor *Calidad* es percibida por los expertos (correlación positiva). Mientras que los altos niveles de *Acidez Volátil* (olor a vinagre) desplomaron la calificación (correlación negativa).

---

## 👩‍💻 Autoría
**Vanessa Cortes**
*Programación de APIs e Inteligencia Artificial Aplicada (2026).*