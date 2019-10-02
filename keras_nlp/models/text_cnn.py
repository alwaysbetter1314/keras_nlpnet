from keras.layers import concatenate,Reshape,Conv2D,MaxPool2D,Input

a:str =1
print(type(a) )

class TextCNN():
  '''
    瞎几把实现，反正没有xlnet和bert吊
  def TextCNN(max_features, maxlen=50,nclass=15, embed_size=256, hid_size= 192, pool_size=50 ):
  '''
  def __init__(self,max_features, nclass, maxlen,embed_size):
    self.input_tensor = Input(shape=(maxlen,))
    self.max_features = max_features
    self.maxlen = maxlen
    self.embed_size = embed_size

  def model(self):
    x = Embedding(self.max_features, self.embed_size)(self.input_tensor)
    
    conv1 = Conv1D(filters=64, kernel_size=1, padding='same')(x)
    conv1 = MaxPool1D(pool_size=32)(conv1)
    
    
    conv2 = Conv1D(filters=64, kernel_size=2, padding='same')(x)
    conv2 = MaxPool1D(pool_size=32)(conv2)
    
    conv3 = Conv1D(filters=64, kernel_size=3, padding='same')(x)
    conv3 = MaxPool1D(pool_size=32)(conv3)
    
    conv4 = Conv1D(filters=64, kernel_size=4, padding='same')(x)
    conv4 = MaxPool1D(pool_size=32)(conv4)
    
    cnn = concatenate([conv1, conv2, conv3, conv4], axis=-1)
    flat = Flatten()(cnn)

    x = Dense(50, activation="relu")(flat)
    x = Dropout(0.1)(x)
    x = Dense(self.nclass, activation="softmax")(x)
    model = Model(inputs=self.input_tensor , outputs=x)
    model.summary()
    return model