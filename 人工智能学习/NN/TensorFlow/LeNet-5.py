import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
from tensorflow.python.keras.datasets.cifar import load_batch

def preprocess(x, y):
    # [0~1]
    x = 2*tf.cast(x, dtype=tf.float32) / 255.-1
    y = tf.cast(y, dtype=tf.int32)
    return x,y

def load_data(path="TensorFlow/data/mnist.npz"):
    if path=="TensorFlow/data/mnist.npz":
        f = np.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return (x_train, y_train), (x_test, y_test)
    elif path=="TensorFlow/data/cifar-10-batches-py/":
        num_train_samples = 50000
        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')
        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
            y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        
        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)

        return (x_train, y_train), (x_test, y_test)

(x, y), (x_test, y_test) = load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32) 
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).batch(200)

val_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
val_db = val_db.shuffle(1000).batch(200)

def main():

    network = Sequential([
        layers.Conv2D(6, kernel_size=3, strides=1),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Conv2D(16, kernel_size=3, strides=1),
        layers.MaxPooling2D(pool_size=2,strides=2),
        layers.ReLU(),
        layers.Flatten(),

        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10)
    ])
    network.build(input_shape=(4, 28, 28, 1))
    network.summary()

    optimizer = optimizers.Adam(lr=1e-4)
    criteon = losses.CategoricalCrossentropy(from_logits=True)
    variables = network.trainable_variables

    for epoch in range(30):
        for step, (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                x = tf.expand_dims(x, axis=3)
                out = network(x)
                y_onthot = tf.one_hot(y, depth=10)
                loss = criteon(y_onthot, out)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
        
        correct, total = 0, 0
        for x,y in val_db:
            x = tf.expand_dims(x, axis=3)
            out = network(x)
            pred = tf.argmax(out, axis=-1)
            y = tf.cast(y, tf.int64)
            correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y),tf.float32)))
            total += x.shape[0]
        print(epoch, 'test acc:', correct/total)


if __name__ == '__main__':
    main()