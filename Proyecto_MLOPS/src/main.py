
import numpy as np
import tensorflow as tf
import logging
import wandb
from model import build_model
from train import train_model
from utils import prepare_data, evaluate_classifier, plot_training_history_classification, load_config
from logging_config import setup_logging


def main():

    # =========================================
    # PREPARACIÓN
    # =========================================
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    setup_logging("info")
    logger = logging.getLogger(__name__)
    logger.info("Empezamos la ejecucion.")

    config = load_config("general.yaml")
    data = prepare_data('./Data/features_30_sec.csv', seed=42)


    wandb.init(project = "MLOPS_Project", config=config, job_type="training", name = "prueba_patience5_batch_size64", mode = "disabled")  # Si se quiere NO USAR wandb sin borrar código, añadir mode="disabled".
                                                                                                                                          # Esto desactiva completamente W&B: no se registran métricas, artefactos ni logs.
                                                                                                                                          # Las llamadas a wandb (callbacks, log_artifact, etc.) se ignoran silenciosamente,
                                                                                                                                          # por lo que no afectan ni interrumpen la ejecución del código.   
    
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]

    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    input_dim = data["input_dim"]
    num_classes = data["num_classes"]


    # =========================================
    # ENTRENAMIENTO
    # =========================================

    model_without_training = build_model(input_dim, num_classes)
    model, history = train_model(
        model_without_training,
        X_train, y_train,
        X_val, y_val,
        config["training"]["epochs"],
        config["training"]["batch_size"],
        config["training"]["shuffle"],
        config["early_stopping"]["patience"],
        config["early_stopping"]["monitor"],
        config["model"]["path"]
    )

    # =========================================
    # EVALUACIÓN
    # =========================================

    class_names = data.get("label_encoder").classes_
    evaluate_classifier(model, X_test, y_test, class_names, "Music Genre")
    plot_training_history_classification(history)

    logger.info("EJECUCION TERMINADA.\n")
    logger.info("=" * 50)
    logger.info("-" * 50)
    logger.info("==================================================\n")



if __name__ == "__main__":
    main()