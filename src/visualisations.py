import math
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def plot_image_examples(images : tf.Tensor, n=9):
    rows = int(math.sqrt(n))
    fig, axs = plt.subplots(rows, math.ceil(n / rows))
    for i in range(n):
        axs[i // rows, i % rows].imshow(images[i], cmap="gray")
    fig.show()
    plt.show()

def plot_changed_images(images : tf.Tensor, noised_image : tf.Tensor, changed_images : tf.Tensor, n=8):
    fig, axs = plt.subplots(3, n)
    for i in range(n):
        axs[0, i].imshow(images[i], cmap="gray")
        axs[1, i].imshow(noised_image[i], cmap="gray")
        axs[2, i].imshow(changed_images[i], cmap="gray")
    fig.show()
    plt.show()
