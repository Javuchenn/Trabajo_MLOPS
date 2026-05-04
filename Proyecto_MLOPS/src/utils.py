import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.utils import to_categorical

from pathlib import Path
import yaml

import logging
logger = logging.getLogger(__name__)


# =========================================
# ARGUMENTOS DE ENTRADA CON YAML
# =========================================

def get_project_folder() -> Path:
    raiz_proyecto = Path(__file__).resolve().parents[1]
    return raiz_proyecto

def load_config(filename: str) -> dict:
    logger.info("Cargando la configuracion.")
    fichero_configuracion = get_project_folder() / "config" / filename
    with open(fichero_configuracion, "r") as fichero:
        return yaml.safe_load(fichero)




# =========================================
# VISUALIZACIÓN DEL HISTORIAL DE ENTRENAMIENTO
# =========================================

def plot_training_history_classification(history, title="Training History"):
    """Plot training and validation metrics for classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Categorical Crossentropy')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    print("\n\n")

def evaluate_classifier(model, X_test, y_test, class_names, model_name="Model"):
    """Comprehensive classifier evaluation."""
    pred_probs = model.predict(X_test, verbose=0)
    pred = np.argmax(pred_probs, axis=1)

    # 🔧 FIX MÍNIMO: convertir one-hot a clases
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test, pred)

    logger.info("Iniciando la evaluacion...")

    logger.info(f"{'='*60}")
    logger.info(f"{model_name} Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"{'='*60}\n")
    logger.info(classification_report(y_test, pred, target_names=class_names))

    return {'Accuracy': acc, 'Predictions': pred, 'Probabilities': pred_probs}




# =========================================
# FUNCIONES DE CARGA Y PREPROCESADO DE DATOS
# =========================================
def load_data(path):
    return pd.read_csv(path)


def clean_dataframe(df, target_col):
    df = df.dropna(subset=[target_col])

    feature_cols = [c for c in df.columns if c not in ['filename', 'label', 'genre']]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    return df, feature_cols


def split_features_target(df, feature_cols, target_col):
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    return y_encoded, le, num_classes


def split_data(X, y, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=seed,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler

def one_hot_encode(y_train, y_val, y_test, num_classes):
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val   = to_categorical(y_val, num_classes=num_classes)
    y_test  = to_categorical(y_test, num_classes=num_classes)

    return y_train, y_val, y_test

def prepare_data(path, seed=42):

    logger.info("Cargando el dataset.")

    df = load_data(path)
    target_col = 'label' if 'label' in df.columns else 'genre'
    df, feature_cols = clean_dataframe(df, target_col)
    X, y = split_features_target(df, feature_cols, target_col)
    y, le, num_classes = encode_labels(y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)
    y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test, num_classes)
    input_dim = X_train.shape[1]

    logger.info("Dataset cargado.")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "label_encoder": le,
        "scaler": scaler,
        "feature_cols": feature_cols
    }