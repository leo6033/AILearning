import os
import numpy as np
import glob
import tensorflow as tf
from scipy.misc import toimage
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from util import make_anime_dataset

class Generator(keras.Model):
    # 生成器网络
    def __init__(self):
        super(Generator, self).__init__()
        filter = 64
        # 转置卷积层1，输出channel为filter*8，核大小4，步长1，不使用padding，不使用偏置
        self.conv1 = layers.Conv2DTranspose(filter*8, 4, 1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 转置卷积层2
        self.conv2 = layers.Conv2DTranspose(filter*4, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 转置卷积层3
        self.conv3 = layers.Conv2DTranspose(filter*2, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 转置卷积层4
        self.conv4 = layers.Conv2DTranspose(filter*1, 4, 2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 转置卷积层5
        self.conv5 = layers.Conv2DTranspose(3, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs # [z, 100]
        # Reshape 成 4D 张量，方便后续转置卷积运算：(b, 1, 1, 100)
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        x = tf.nn.relu(x)
        # 转置卷积-BN-激活函数:(b, 4, 4, 512)
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        # 转置卷积-BN-激活函数:(b, 8, 8, 256)
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        # 转置卷积-BN-激活函数:(b, 16, 16, 128)
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        # 转置卷积-BN-激活函数:(b, 32, 32, 64)
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        # 转置卷积-激活函数:(b, 64, 64, 3)
        x = self.conv5(x)
        x = tf.tanh(x) # 输出 x 范围 -1~1，与预处理一致

        return x
    
class Discriminator(keras.Model):
    # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()
        filter = 64
        # 卷积层1
        self.conv1 = layers.Conv2D(filter, 4, 2, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 卷积层2
        self.conv2 = layers.Conv2D(filter*2, 4, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 卷积层3
        self.conv3 = layers.Conv2D(filter*4, 4, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 卷积层4
        self.conv4 = layers.Conv2D(filter*8, 3, 1, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 卷积层5
        self.conv5 = layers.Conv2D(filter*16, 3, 1, 'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        # 全局池化层
        self.pool = layers.GlobalAveragePooling2D()
        # 特征打平层
        self.flatten = layers.Flatten()
        # 2分类全连接层
        self.fc = layers.Dense(1)
    
    def call(self, inputs, training=None):
        # 卷积-BN-激活函数:(4, 31, 31, 64)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        # 卷积-BN-激活函数:(4, 14, 14, 128)
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        # 卷积-BN-激活函数:(4, 6, 6, 256)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        # 卷积-BN-激活函数:(4, 4, 4, 512)
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        # 卷积-BN-激活函数:(4, 2 ,2, 1024)
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        # 池化:(4, 1024)
        x = self.pool(x)
        # 打平
        x = self.flatten(x)
        # 输出，[b, 1024] => [b, 1]
        logits = self.fc(x)

        return logits

def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    # 计算属于与标签为0的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算判别器的误差函数
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 判定生成图片
    d_fake_logits = discriminator(fake_image, is_training)
    # 判定真实图片
    d_real_logits = discriminator(batch_x, is_training)
    # 真实图片与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss

def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)
    return loss

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)

def main():
    
    tf.random.set_seed(3333)
    np.random.seed(3333)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')


    z_dim = 100 # 隐藏向量z的长度
    epochs = 3000000 # 训练步数
    batch_size = 64 # batch size
    learning_rate = 0.0002
    is_training = True

    # 获取数据集路径
    # C:\Users\z390\Downloads\anime-faces
    # r'C:\Users\z390\Downloads\faces\*.jpg'
    img_path = glob.glob(r'TensorFlow\data\faces\*.jpg') + \
        glob.glob(r'TensorFlow\data\faces\*.png')
    # img_path = glob.glob(r'C:\Users\z390\Downloads\getchu_aligned_with_label\GetChu_aligned2\*.jpg')
    # img_path.extend(img_path2)
    print('images num:', len(img_path))
    # 构建数据集对象
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)
    print(dataset, img_shape)
    sample = next(iter(dataset)) # 采样
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat(100) # 重复循环
    db_iter = iter(dataset)


    generator = Generator() # 创建生成器
    generator.build(input_shape = (4, z_dim))
    discriminator = Discriminator() # 创建判别器
    discriminator.build(input_shape=(4, 64, 64, 3))
    # 分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    # generator.load_weights('generator.ckpt')
    # discriminator.load_weights('discriminator.ckpt')
    # print('Loaded chpt!!')

    d_losses, g_losses = [],[]
    for epoch in range(epochs): # 训练epochs次
        # 1. 训练判别器
        for _ in range(1):
            # 采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter) # 采样真实图片
            # 判别器前向计算
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        # 2. 训练生成器
        # 采样隐藏向量
        batch_z = tf.random.normal([batch_size, z_dim])
        batch_x = next(db_iter) # 采样真实图片
        # 生成器前向计算
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss))
            # 可视化
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('TensorFlow\gan_images', 'gan-%d.png'%epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

            if epoch % 10000 == 1:
                # print(d_losses)
                # print(g_losses)
                generator.save_weights('generator.ckpt')
                discriminator.save_weights('discriminator.ckpt')

if __name__ == '__main__':
    main()