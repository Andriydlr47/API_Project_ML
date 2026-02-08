import os
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Cargar variables de entorno

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise RuntimeError(
        "Las variables de entorno DATABRICKS_HOST y DATABRICKS_TOKEN no están definidas"
    )

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# Configurar MLflow y cargar modelo

mlflow.set_tracking_uri("databricks")
MODEL_URI = "models:/workspace.default.energia_tenerife_rf/1"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo desde MLflow: {e}")


# Esquema de entrada

class EnergiaInput(BaseModel):
    instalacion_id: int
    energia_importada: float
    velocidad_viento_media: float
    radiacion_solar_total: float
    mes_seno: float
    mes_coseno: float
    anio: int


# Inicializar FastAPI

app = FastAPI(
    title="API Predicción Energía Exportada Tenerife",
    description="Modelo ML entrenado con datos energéticos y meteorológicos de Canarias",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o tu dominio frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints

@app.get("/health")
def health():
    """Endpoint de salud de la API"""
    return {
        "status": "ok",
        "model_uri": MODEL_URI
    }

@app.post("/predict")
def predict(data: EnergiaInput):
    """
    Endpoint de predicción.
    Recibe las variables de entrada y devuelve la energía exportada estimada.
    """
    try:
        # Convertir entrada a DataFrame
        df = pd.DataFrame([data.dict()])

        # Predicción
        prediction = model.predict(df)[0]

        return {
            "energia_exportada_predicha": float(prediction)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(e)}"
        )



