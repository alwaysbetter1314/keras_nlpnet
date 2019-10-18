import numpy as np
from keras.preprocessing import sequence
from keras_nlp.models import text_cnn
from keras.datasets import imdb
from keras.utils import to_categorical

#params
max_features = 20000
maxlen = 100
batch_size = 32
num_classes = 2

# load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = to_categorical(y_train,num_classes=num_classes)
y_test = to_categorical(y_test,num_classes=num_classes)



# train
model = text_cnn(num_classes=num_classes,
	act='sigmoid',
	max_features=max_features,
	maxlen=maxlen )
model.summary()
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit( x_train,
		   y_train,
          epochs=4,
          validation_data=[x_test,y_test]

          )