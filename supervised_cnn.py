import tensorflow as tf
import numpy as np
from dataset import get_train_test_split
from tensorflow.keras import Model
import time
#batch_size = 64

class CNN(Model):
    def __init__(self, input_shape = (28, 28, 1), output_size = 10):
        super(CNN, self).__init__()

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPool2D(),  # Pooling layer to downsample
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),  # Pooling layer to downsample
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax')
        ])
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        print(self.model.summary())


    def call(self, x):
       return self.model(x)
    
    def train(self, train_ds, test_ds=None, epochs = 3):
        start_time = time.time()
        for epoch in range(epochs):
            print(f"\nStart of epoch {(epoch+1)}/{epochs}")
            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                if (step+1)%100==0:
                    if (step+1)%1000==0:
                        print(f"Step: {step+1}. Time: {(time.time() - start_time)}")
                    else:
                        print(f"Step: {step+1}.")
                
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)  
                    y_batch_train = tf.reshape(y_batch_train, logits.shape)
                    loss_value = self.loss_fn(y_batch_train, logits)

                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                self.train_accuracy.update_state(y_batch_train, logits)
                self.train_accuracy.reset_states()
                if test_ds: 
                     self.test(test_ds)
                     if step%10==0 and step!=0:
                        test_acc = self.test_accuracy.result()
                        print("Test acc: %.4f" % (float(test_acc),))


    def test(self,test_ds):
        for x_batch_test, y_batch_test in test_ds:
            predictions = self.model(x_batch_test)
            self.test_accuracy.update_state(y_batch_test, predictions)
        self.test_accuracy.reset_states()

    def save(self,name):
        self.model.save(name)

def load_model(name):
    loaded_model = tf.keras.models.load_model(name)
    cnn = CNN()
    cnn.model = loaded_model
    return cnn