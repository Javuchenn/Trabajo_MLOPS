
import numpy as np
import tensorflow as tf
import logging
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

    setup_logging("debug")
    logger = logging.getLogger(__name__)
    logger.info("Empezamos la ejecución.")

    config = load_config("config.yaml")
    data = prepare_data('./Data/features_30_sec.csv', seed=42)

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

    logger.info("\n\nEJECUCIÓN TERMINADA.\n")
    logger.info("=" * 50)
    logger.info("-" * 50)
    logger.info("=" * 50)
    logger.info("\n")



if __name__ == "__main__":
    main()