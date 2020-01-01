import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential

def load_data(path='TensorFlow/data/imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              **kwargs):

  with np.load(path, allow_pickle=True) as f:
    x_train, labels_train = f['x_train'], f['y_train']
    x_test, labels_test = f['x_test'], f['y_test']

  np.random.seed(seed)
  indices = np.arange(len(x_train))
  np.random.shuffle(indices)
  x_train = x_train[indices]
  labels_train = labels_train[indices]

  indices = np.arange(len(x_test))
  np.random.shuffle(indices)
  x_test = x_test[indices]
  labels_test = labels_test[indices]

  xs = np.concatenate([x_train, x_test])
  labels = np.concatenate([labels_train, labels_test])

  if start_char is not None:
    xs = [[start_char] + [w + index_from for w in x] for x in xs]
  elif index_from:
    xs = [[w + index_from for w in x] for x in xs]

  if maxlen:
    xs, labels = _remove_long_seq(maxlen, xs, labels)
    if not xs:
      raise ValueError('After filtering for sequences shorter than maxlen=' +
                       str(maxlen) + ', no sequence was kept. '
                       'Increase maxlen.')
  if not num_words:
    num_words = max([max(x) for x in xs])

  # by convention, use 2 as OOV word
  # reserve 'index_from' (=3 by default) characters:
  # 0 (padding), 1 (start), 2 (OOV)
  if oov_char is not None:
    xs = [
        [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
    ]
  else:
    xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

  idx = len(x_train)
  x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
  x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

  return (x_train, y_train), (x_test, y_test)

def get_word_index(path='TensorFlow/data/imdb_word_index.json'):

  with open(path) as f:
    return json.load(f)

batch_size = 128
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = load_data(num_words=total_words)

print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)
print(x_train[0])

word_index = get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(x_train[0]))

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# 构建数据集，打散，批量，并丢掉最后一个不够batchsize的batch
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

class MyRNN(keras.Model):
    # Cell方式构建多层网络
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # [b, 64]，构建Cell初始化状态向量，重复使用
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]
        # 词向量编码 [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        # 构建2个Cell
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        # # 构建RNN
        # self.rnn = keras.Sequential([
        #     layers.SimpleRNN(units, dropout=0.5, return_sequences=True),
        #     layers.SimpleRNN(units, dropout=0.5)
        # ])
        # 构建分类网络，用于将CELL的输出特征进行分类，2分类
        # [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = Sequential([
        	layers.Dense(units),
        	layers.Dropout(rate=0.5),
        	layers.ReLU(),
        	layers.Dense(1)])

    def call(self, inputs, training=None):
        x = inputs # [b, 80]
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute,[b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1): # word: [b, 100] 
            out0, state0 = self.rnn_cell0(word, state0, training) 
            out1, state1 = self.rnn_cell1(out0, state1, training)
        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        x = self.outlayer(out1, training)
        # # rnn cell compute,[b, 80, 100] => [b, 64]
        # x = self.rnn(x)
        # # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        # x = self.outlayer(x,training)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob

def main():
    units = 64 # RNN状态向量长度f
    epochs = 50 # 训练epochs

    model = MyRNN(units)
    # 装配
    model.compile(optimizer = optimizers.RMSprop(0.001),
                  loss = losses.BinaryCrossentropy(),
                  metrics=['accuracy'],
                  run_eagerly=True)
    # 训练和验证
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    # 测试
    model.evaluate(db_test)


if __name__ == '__main__':
    main()