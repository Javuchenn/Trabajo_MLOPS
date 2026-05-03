import tensorflow as tf
from src.model import build_model


def test_model_build():
    model = build_model(input_dim=10, num_classes=3)

    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == 3