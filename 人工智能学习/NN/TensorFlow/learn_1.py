import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets

x = tf.range(9)
print(x[:-1])

# [b, h, w, c] --> [b, c, h, w]
# 数量 高 宽 通道  -->  数量 通道 高 宽
# tf.transpose(x, perm=[0, 3, 1, 2])

'''

'''

"""
# 设置GPU显存占用为按需分配
# 插入到模型创建前
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_physical_devices('GPU')
        print(len(gpus),  "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
"""
