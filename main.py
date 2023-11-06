
from dataset import get_train_test_split, preprocess_data
from supervised_cnn import CNN
import tensorflow_datasets as tfds


ds_train, ds_test = get_train_test_split()


ds_train_preprocessed = ds_train.map(preprocess_data)
ds_test_preprocessed = ds_test.map(preprocess_data)
cnn = CNN(output_size=10)

cnn.train(train_ds=ds_train_preprocessed, test_ds=None, epochs=3)
cnn.save("my_first_model")
