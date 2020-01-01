import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    # 残差模块
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积单元
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二个卷积单元
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1: # 通过 1x1 卷积完成shape匹配
            self.down_sample = Sequential()
            self.down_sample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else: # shape 匹配，直接短接
            self.downsample = lambda x: x
        
    def call(self, inputs, training=None):
        # [b, h, w, c], 通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        # 通过identity模块
        identity = self.down_sample(inputs)
        # 2条路径输出直接相加
        output = layers.add([out, identity])
        output = tf.nn.relu(output) # 激活函数

        return output

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 根网络， 预处理
        self.stem = Sequential([layers.Conv2D(64, (3, 3),strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 堆叠4个Block，每个block包含了多个BasicBlock，设置步长不一样
        self.layer1 = self.bulid_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # 通过polling层将高度降低为1x1
        slef.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)
    
    def call(self, inputs, training=None):
        # 通过根网络
        x = self.stem(inputs)
        # 四个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 通过池化层
        x = self.avgpool(x)
        # 通过全连接层
        x = self.fx(x)

        return x

    def build_resblock(self, filter_num, blocks, strides=1):
        # 辅助函数，堆叠filter_num个BasicBlock
        res_blocks = Sequential()
        # 只有第一个BasicBlock的不上可能不为1，实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks): # 其他BasicBlock步长都为1
            res_blocks.add(BasicBlock(filter_num, stride=1))
        
        return res_blocks

def resnet18(): 
    # 通过调整模块内部 BasicBlock 的数量和配置实现不同的 ResNet     
    return ResNet([2, 2, 2, 2])

def resnet34():
    return ResNet([3, 4, 6, 3])
