import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Model
def text_cnn(   num_classes=10, 
                max_features=20000, 
                maxlen= 50,
                embed_size =128, 
                hid_dim_conv = 64,
                pool_size = 32,
                hid_dim_dense = 50,
                dropout_rate = 0.1,
                act='softmax'):
    input_tensor = L.Input(shape=(maxlen,), dtype='int32')
    emb1 = L.Embedding(max_features, embed_size)(input_tensor)
    conv1 = L.Conv1D(filters=hid_dim_conv, kernel_size=1, padding='same')(emb1)
    pool1 = L.MaxPool1D(pool_size=pool_size)(conv1)
    conv2 = L.Conv1D(filters=hid_dim_conv, kernel_size=2, padding='same')(emb1)
    pool2 = L.MaxPool1D(pool_size=pool_size)(conv2)
    conv3 = L.Conv1D(filters=hid_dim_conv, kernel_size=3, padding='same')(emb1)
    pool3 = L.MaxPool1D(pool_size=pool_size)(conv3)
    conv4 = L.Conv1D(filters=hid_dim_conv, kernel_size=4, padding='same')(emb1)
    pool4 = L.MaxPool1D(pool_size=pool_size)(conv4)
    concat = L.concatenate([conv1, conv2, conv3, conv4], axis=-1)
    flat = L.Flatten()(concat)
    dense1 = L.Dense(hid_dim_dense, activation="relu")(flat)
    drop1 = L.Dropout(dropout_rate)(dense1)
    output = L.Dense(num_classes,activation=act)(drop1)
    return Model(inputs = input_tensor, outputs = output)


if __name__ == "__main__":
    model = text_cnn()
    model.summary()  


# class TextCNN(tf.keras.Model):
#       '''
#       init ： 定义网络层
#       call : 定义前向传播
#       '''

#       def __init__(self, num_classes=10, 
#                     max_features=20000, 
#                     maxlen= 50,
#                     embed_size =128, 
#                     hid_dim_conv = 64,
#                     pool_size = 32,
#                     hid_dim_dense = 50,
#                     dropout_rate = 0.1,
#                     act='softmax',

#                     ):
#         super(TextCNN, self).__init__(name='my_textcnn_model')
#         self.num_classes = num_classes
#         self.max_features = max_features
#         self.embed_size = embed_size

#         self.emb1 = L.Embedding(self.max_features, self.embed_size)
#         self.conv1 = L.Conv1D(filters=hid_dim_conv, kernel_size=1, padding='same')
#         self.pool1 = L.MaxPool1D(pool_size=pool_size)
#         self.conv2 = L.Conv1D(filters=hid_dim_conv, kernel_size=2, padding='same')
#         self.pool2 = L.MaxPool1D(pool_size=pool_size)
#         self.conv3 = L.Conv1D(filters=hid_dim_conv, kernel_size=3, padding='same')
#         self.pool3 = L.MaxPool1D(pool_size=pool_size)
#         self.conv4 = L.Conv1D(filters=hid_dim_conv, kernel_size=4, padding='same')
#         self.pool4 = L.MaxPool1D(pool_size=pool_size)
#         self.flat = L.Flatten()
#         self.dense1 = L.Dense(hid_dim_dense, activation="relu")
#         self.drop1 = L.Dropout(dropout_rate)
#         self.dense2 = L.Dense(self.num_classes,activation=act)

#         self.build(input_shape=(None,maxlen))

#       def call(self, inputs):
#         x = self.emb1(inputs)
#         conv1 = self.conv1(x)
#         conv1 = self.pool1(conv1)
#         conv2 = self.conv2(x)
#         conv2 = self.pool2(conv2)
#         conv3 = self.conv3(x)
#         conv3 = self.pool3(conv3)
#         conv4 = self.conv4(x)
#         conv4 = self.pool4(conv4)

#         concat = L.concatenate([conv1, conv2, conv3, conv4], axis=-1)
#         flat = self.flat(concat)

#         x = self.dense1(flat)
#         x = self.drop1(x)
#         return self.dense2(x)

#       def compute_output_shape(self, input_shape):

#         # 如果想要使用定义的子类，则需要重载此方法，计算模型输出的形状
#         shape = tf.TensorShape(input_shape).as_list()
#         shape[-1] = self.num_classes
  


    