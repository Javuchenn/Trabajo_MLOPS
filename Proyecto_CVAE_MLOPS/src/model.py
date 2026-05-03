import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
import logging

logger = logging.getLogger(__name__)

def build_model(input_dim, num_classes):
    logger.info("Construyendo modelo...")
    model = Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(
            units=128,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None,
            activity_regularizer=None
        ),

        layers.Dense(
            units=64,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None,
            activity_regularizer=None
        ),

        layers.Dense(
            units=32,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None,
            activity_regularizer=None
        ),

        layers.Dense(
            units=16,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None,
            activity_regularizer=None
        ),

        layers.Dense(
            units=num_classes,
            activation='softmax',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=None,
            activity_regularizer=None
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("Modelo construido.")
    
    return model
