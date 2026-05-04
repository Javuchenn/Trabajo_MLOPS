import tensorflow as tf
import logging
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import get_project_folder

logger = logging.getLogger(__name__)


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, shuffle, early_stopping_patience, early_stopping_monitor, model_path):

    logger.info("Iniciando entrenamiento.")
    logger.info(f"Epochs: {epochs} | Batch size: {batch_size} | Patience: {early_stopping_patience}")
    
    early_stop = EarlyStopping(
        monitor=early_stopping_monitor,
        patience=early_stopping_patience,
        restore_best_weights=True            # Con esto el modelo final queda con los pesos del epoch que tuvo mejor val_loss
    )

    checkpoint = ModelCheckpoint(
        model_path,                          # Se guarda el mejor modelo
        monitor=early_stopping_monitor,
        save_best_only=True,
        mode='min',
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=1,
        callbacks=[early_stop, checkpoint, WandbMetricsLogger()]
    )


    model_artifact = wandb.Artifact(
        name="modelo_entrenado",
        type="model",
        description="Modelo ya entrenado"
    )
    raiz = get_project_folder()
    model_artifact.add_file(raiz/"models"/"best_model.keras")
    wandb.log_artifact(model_artifact)


    logger.info("Entrenamiento finalizado.")

    return model, history