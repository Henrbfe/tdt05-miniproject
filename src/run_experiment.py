import argparse
import tensorflow as tf

from data import load_data, add_image_noise
from autoencoder import get_trained_autoencoder
from models import train_supervised_model, evaluate_model
from visualisations import plot_changed_images




def run_experiment():
    """ Evaluate the usefulness of an autoencoder to improve a simple mlp-classification task. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-lpor", "--LabeledPortion", help = "Provide a fraction of the data that should contain labels and be used for supervised learning. Default is 0.2")
    parser.add_argument("-lsd", "--LatentSpaceDimensions", help = "Provide the dimensions of the latens space used by the encoder. Default 64")
    parser.add_argument("-viz", "--Visualize", help = "Set to True if visualizations should be included during experiment.")
    args = parser.parse_args()

    labeled_split = args.LabeledPortion if args.LabeledPortion else 0.01
    visualize = args.Visualize

    (x_train, noisy_x_train,
    labeled_x_train, y_train,
    x_test, noisy_x_test, y_test) = load_data(tf.keras.datasets.fashion_mnist, labeled_split, visualize=visualize)

    autoencoder_noisy = get_trained_autoencoder(noisy_x_train, x_train, latent_dim=64, input_shape=(28, 28, 1), filename="model_noisy_v1.1")
    autoencoder = get_trained_autoencoder(x_train, x_train, latent_dim=64, input_shape=(28, 28, 1), filename="model_v1.1")

    if visualize:
        plot_changed_images(x_test, x_test, autoencoder(x_test))
        plot_changed_images(x_test, noisy_x_test, autoencoder(noisy_x_test))
        plot_changed_images(x_test, noisy_x_test, autoencoder_noisy(noisy_x_test))

    supervised_x = train_supervised_model((28, 28, 1), 10, labeled_x_train, y_train)
    encdec_x = train_supervised_model((28, 28, 1), 10, autoencoder(labeled_x_train), y_train)
    enc_x = train_supervised_model((7, 7, 10), 10, autoencoder.encoder(labeled_x_train), y_train)

    supervised_noisy = train_supervised_model((28, 28, 1), 10, add_image_noise(labeled_x_train), y_train)
    encdec_noisy = train_supervised_model((28, 28, 1), 10, autoencoder_noisy(add_image_noise(labeled_x_train)), y_train)
    enc_noisy = train_supervised_model((7, 7, 10), 10, autoencoder_noisy.encoder(add_image_noise(labeled_x_train)), y_train)

    evaluate_model(supervised_x, x_test, y_test, model_name="supervised_x")
    evaluate_model(supervised_x, noisy_x_test, y_test, model_name="supervised_x_noisy")
    evaluate_model(supervised_noisy, noisy_x_test, y_test, model_name="supervised_noisy")

    evaluate_model(encdec_x, autoencoder(x_test), y_test, model_name="encdec_x")
    evaluate_model(encdec_x, autoencoder(noisy_x_test), y_test, model_name="encdec_x_noisy")
    evaluate_model(encdec_noisy, autoencoder_noisy(noisy_x_test), y_test, model_name="encdec_noisy")

    evaluate_model(enc_x, autoencoder.encoder(x_test), y_test, model_name="enc_x")
    evaluate_model(enc_x, autoencoder.encoder(noisy_x_test), y_test, model_name="enc_x_noisy")
    evaluate_model(enc_noisy, autoencoder_noisy.encoder(noisy_x_test), y_test, model_name="enc_noisy")


if __name__ == "__main__":
    run_experiment()
