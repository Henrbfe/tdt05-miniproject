import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from constants import CLASS_NAMES


# Construct a tf.data.Dataset
ds_train, ds_test = tfds.load('fashion_mnist', split=['train','test'], shuffle_files=False)

#ds_train = ds_train.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)



def get_train_test_split(visualise=False):
    ds_train, ds_test = tfds.load('fashion_mnist', split=['train','test'], shuffle_files=False)
    if visualise: 
        plt.figure(figsize=(10, 10))
        i = 0
        for example in ds_train.take(36):
            image = example['image']  # Extract the image
            label = example['label']  # Extract the label

            ax = plt.subplot(6, 6, i + 1)
            plt.imshow(image.numpy().reshape(28, 28), cmap='gray')
            plt.title(f"Label: {label.numpy()}, {CLASS_NAMES[int(label.numpy())]}")
            plt.axis('off')
            i+=1
        plt.show()
    return ds_train, ds_test
    
