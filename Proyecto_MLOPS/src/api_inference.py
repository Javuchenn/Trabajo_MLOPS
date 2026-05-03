
"""
API de Clasificación de Géneros Musicales

Este servicio recibe un vector de características que representa una muestra de audio
(todas las variables numéricas ya preprocesadas usadas durante el entrenamiento del modelo).

Flujo de funcionamiento:
- La API espera una lista de floats ("features") en el mismo orden que en el entrenamiento.
- Estas características se pasan directamente al modelo de TensorFlow.
- El modelo devuelve una distribución de probabilidades para cada género musical.
- Se selecciona la clase con mayor probabilidad (argmax) como predicción.
- La predicción numérica se traduce a un nombre de género legible.
- La API devuelve:
    - ID de la clase predicha
    - nombre del género predicho
    - probabilidades para todas las clases
    - nivel de confianza (probabilidad máxima)

Formato exacto de entrada (OBLIGATORIO):
- Tipo: list[float]
- Dimensión: exactamente INPUT_DIM valores (mismo número de features que en entrenamiento)
- Orden: estrictamente el mismo orden de columnas usado en el dataset de entrenamiento
- Escalado: ya deben estar normalizadas con StandardScaler (NO enviar valores en crudo)
- Codificación: no se deben incluir labels ni strings, solo valores numéricos
- Forma final esperada:
    [f1, f2, f3, ..., fN]  → donde N = número de features del modelo

Features utilizadas en entrenamiento:
El modelo fue entrenado con un dataset de audio donde cada fila contiene características extraídas de señales de audio:
- chroma_stft_mean / chroma_stft_var
- rms_mean / rms_var
- spectral_centroid_mean / spectral_centroid_var
- spectral_bandwidth_mean / spectral_bandwidth_var
- rolloff_mean / rolloff_var
- zero_crossing_rate_mean / zero_crossing_rate_var
- harmony_mean / harmony_var
- perceptr_mean / perceptr_var
- tempo
- mfcc1_mean ... mfcc20_mean
- mfcc1_var ... mfcc20_var
- length

Importante:
La API NO realiza extracción ni preprocesado de características.
La entrada debe coincidir exactamente con el formato usado en el entrenamiento.
"""

import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.utils import load_config, get_project_folder

# =========================================
# RUTAS
# =========================================
PROJECT_ROOT = get_project_folder()
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.h5"


# =========================================
# INPUT DE LA API
# =========================================
class MusicInput(BaseModel):
    features: list[float]


# =========================================
# CICLO DE VIDA
# =========================================
@asynccontextmanager
async def lifespan(app: FastAPI):

    # cargar modelo entrenado
    model = tf.keras.models.load_model(MODEL_PATH)

    app.state.model = model

    # CLASES DEL MODELO (ORDEN FIJO DEL TRAINING)
    app.state.class_names = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]

    yield


# =========================================
# APP
# =========================================
app = FastAPI(
    title="Music Genre Classifier API",
    lifespan=lifespan
)


# =========================================
# ENDPOINT
# =========================================
@app.post("/predict")
def predict(data: MusicInput):

    x = np.array(data.features, dtype=np.float32).reshape(1, -1)

    probs = app.state.model.predict(x, verbose=0)
    pred = int(np.argmax(probs, axis=1)[0])

    # TRADUCCIÓN A LABEL REAL
    label = app.state.class_names[pred]

    return {
        "prediction_class_id": int(pred),
        "prediction_class_name": label,
        "probabilities": probs.tolist()[0],
        "confidence": float(np.max(probs))
    }