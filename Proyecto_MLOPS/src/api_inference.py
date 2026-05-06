
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


##############################
CÓMO PROBAR LA API EN LOCAL:
##############################
Para ejecutar la API en entorno local, se debe lanzar el servidor FastAPI con Uvicorn usando
el comando: $ uvicorn src.api_inference:app --host 0.0.0.0 --port 8000.
Una vez iniciado, la API quedará disponible en http://localhost:8000, donde se puede probar
el endpoint /predict enviando una lista de features en formato JSON. También se puede acceder
a la documentación automática en http://localhost:8000/docs.

Leer el readme del proyecto para mas informacion.

"""


import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from pathlib import Path

# =========================================
# RUTAS
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"

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

    app.state.input_dim = model.input_shape[1]

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

    # VALIDACIÓN CRÍTICA
    if len(data.features) != app.state.input_dim:
        return {
            "error": "Invalid feature size",
            "expected": app.state.input_dim,
            "received": len(data.features)
        }

    x = np.array(data.features, dtype=np.float32).reshape(1, -1)

    # DEBUG SHAPE
    print("INPUT SHAPE:", x.shape)

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
