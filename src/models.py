import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

FAMNIST_LABELS = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

def get_supervised_mlp(input_dim, output_dim):
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_dim),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_dim)
    ])

def train_supervised_model(
        input_dim, output_dim,
        x_train : tf.Tensor, y_train : tf.Tensor
):
    model = get_supervised_mlp(input_dim, output_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(x_train, y_train, epochs=50, validation_split=0.15)
    return model

def evaluate_model(
        trained_model : tf.keras.Model,
        x_test : tf.Tensor,
        y_test : tf.Tensor,
        model_name : str
    ):

    pred = tf.argmax(trained_model(x_test), axis=1).numpy()

    print(f"--------- {model_name} ---------")
    print(classification_report(y_test, pred, target_names=FAMNIST_LABELS))

    return pred
