import numpy as np
from src.model import build_model
from src.train import train_model


def test_train_runs():

    X_train = np.random.rand(20, 10)
    y_train = np.eye(3)[np.random.randint(0, 3, 20)]

    X_val = np.random.rand(5, 10)
    y_val = np.eye(3)[np.random.randint(0, 3, 5)]

    model = build_model(10, 3)

    trained_model, history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=1,
        batch_size=4,
        shuffle=True,
        early_stopping_patience=1,
        early_stopping_monitor="val_loss",
        model_path="test_model.h5"
    )

    assert trained_model is not None
    assert "loss" in history.history