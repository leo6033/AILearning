import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Sequential, losses
from PIL import Image
import matplotlib.pyplot as plt
import gzip

def load_data(path='TensorFlow/data/fashion-mnist/'):

  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
        paths.append(path + fname)

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)

def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)

class AE(keras.Model):
    # 自编码器，包含Encoder和Decoder
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat

h_dim = 20
lr = 1e-3
batch_size = 512

(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batch_size * 5).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batch_size)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = AE()
model.build(input_shape=(None, 784))
model.summary()

optimizer = tf.optimizers.Adam(lr=lr)

for epoch in range(100):
    for step, x in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])
        # 构建梯度记录器
        with tf.GradientTape() as tape:
            # 前向计算获得重建的图片
            x_rec_logits = model(x)
            # 计算重建图片与输入之间的损失函数
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_mean(rec_loss)
        # 自动求导
        grads = tape.gradient(rec_loss, model.trainable_variables)
        # 自动更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

