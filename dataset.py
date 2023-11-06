import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from constants import CLASS_NAMES


def get_train_test_split(visualise=False):
    (ds_train, ds_test), ds_info = tfds.load('fashion_mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)
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
    
# Extract and preprocess data from tf.data.Dataset
def preprocess_data(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (-1, 28, 28, 1))
    label = tf.one_hot(label, depth=10)
    return image, label

# Apply preprocessing to the datasets
# ds_train = ds_train.map(preprocess_data)
# ds_test = ds_test.map(preprocess_data)