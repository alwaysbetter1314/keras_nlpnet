import tensorflow as tf
from tensorflow.keras import layers as L,Model

def fast_text( num_classes=10, 
    max_features = 20000,
    emb_dim =128,
    maxlen= 50, 
    hid_dim= 32,
    act='softmax'):
    input_tensor = L.Input(shape=(maxlen,), dtype='int32')
    emb1  = L.Embedding(max_features, emb_dim)(input_tensor)
    x     = L.GlobalMaxPooling1D()(emb1)
    dense = L.Dense(hid_dim, activation="relu")(x)
    output= L.Dense(num_classes,activation=act)(dense)
    return Model(inputs = input_tensor, outputs = output)
if __name__ == "__main__":
    model = fast_text(10,50)
    model.summary() 

# class FastText(tf.keras.Model):  

# 	def __init__(self, num_classes=10, 
# 						max_features = 20000,
# 						emb_dim =128,
# 						maxlen= 50, 
# 						hid_dim= 32,
# 						act='softmax'):
# 		super(FastText, self).__init__(name='my_fasttext_model')
# 		self.num_classes = num_classes

# 		# 定义这个模型中的层 ，这跟pytorch是类似的
# 		self.emb1    = layers.Embedding(max_features,emb_dim,name="embeding")
# 		self.dense_1 = layers.Dense(hid_dim, activation='relu', name="dense_1")
# 		self.dense_2 = layers.Dense(num_classes, activation=act,name="output")
# 		self.build(input_shape=(maxlen,))

# 	def call(self, input_tensor):

# 		# 定义数据的前向传播，这个和pytorch也是类似的，只不过pytorch是在forward函数中实现的
# 		x = self.emb1(input_tensor)
# 		x = self.dense_1(x)
# 		return self.dense_2(x)

# 	def compute_output_shape(self, input_shape):

# 		# 如果想要使用定义的子类，则需要重载此方法，计算模型输出的形状
# 		shape = tf.TensorShape(input_shape).as_list()
# 		shape[-1] = self.num_classes


 
