import os
import tensorflow as tf


class Autoencoder(tf.keras.Model):
    """ A basic convolutional autoencoder. """
    def __init__(self, latent_dim, shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=shape),
            tf.keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same', strides=2),
            tf.keras.layers.Conv2D(10, (3, 3), activation='relu', padding='same', strides=2),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(10, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_trained_autoencoder(
        x_train : tf.Tensor, y_train : tf.Tensor,
        latent_dim : int, input_shape : tuple[int, int, int],
        filename = "model_v0.1"):

    file_path = f"keras/{filename}.keras"

    if os.path.exists(file_path):
        autoencoder = tf.keras.models.load_model(file_path)
        print("Loaded trained model")
        return autoencoder

    autoencoder = Autoencoder(latent_dim=latent_dim, shape=input_shape)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    autoencoder.fit(x_train, y_train,
                epochs=10,
                shuffle=True,
                validation_split=0.15)
    
    autoencoder.save(file_path)

    return autoencoder
