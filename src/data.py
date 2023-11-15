import numpy as np
import tensorflow as tf
from visualisations import plot_image_examples, plot_changed_images


def add_image_noise(x):
    noise_factor = 0.2
    x_noisy = x + noise_factor * tf.random.normal(shape=x.shape)
    x_noisy = tf.clip_by_value(x_noisy, clip_value_min=0.0, clip_value_max=1.0)
    return x_noisy


def load_data(tf_dataset, labeled_split: float, noise_split=0.5, visualize=False):
    """Loads data from a tensorflow dataset."""
    (x_train, y_train), (x_test, y_test) = tf_dataset.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    if visualize:
        plot_image_examples(x_train)

    y_train, _ = split_data(y_train, labeled_split)
    labeled_x_train = x_train[: len(y_train)]
    noisy_x_test = add_image_noise(x_test)

    noised_x_train = add_image_noise(x_train)

    return (
        x_train,
        noised_x_train,
        labeled_x_train,
        y_train,
        x_test,
        noisy_x_test,
        y_test
    )


def split_data(data: tf.Tensor, portion: float):
    split_index = int(len(data) * portion)
    return data[:split_index], data[split_index:]
