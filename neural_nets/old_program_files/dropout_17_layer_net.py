import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
import math
import timeit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.constraints import max_norm
from PIL import Image
import keras.backend as K

#getting the data
num_training=2400
num_validation=100

X_train = np.zeros((2500,128,128))
y_train = np.zeros((2500,128,128))
for i in range(1,51):
	for j in range(1,51):
		imnoisy = np.array(Image.open('/scratch/user/narendra5/noisy_lines/nline_' +str(i) +'_'+str(j)+'.tiff'))
		im = np.array(Image.open('/scratch/user/narendra5/original_lines/line'+str(i)+'.tiff'))
#		im = scipy.misc.imread(r'$SCRATCH\original_lines\line'+str(i)+'.tiff')
#		imnoisy = scipy.misc.imread(r'$SCRATCH\noisy_lines\nline_' +str(i) +'_'+str(j)+'.tiff')
		X_train[(i-1)*50+j-1] = imnoisy
		y_train[(i-1)*50+j-1] = im


X_train = np.reshape(X_train,(2500,128,128,1))
y_train = np.reshape(y_train,(2500,128,128,1))

X_train = X_train/256
y_train = y_train/256

#print(X_train[0])

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(y_train))

print (X_train.dtype)
#mean_image = np.mean(X_train, axis=0)
#std_image = np.std(X_train, axis=0)


#X_train -= mean_image
#X_val -= mean_image

#X_train /= std_image
#X_val   /= std_image


print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

batch_size = 32
epochs = 100

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=X_train.shape[1:], activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))


#model.add(Dropout(0.5))

model.add(Conv2D(1, (3, 3), padding='same'))


model.summary()


adam = keras.optimizers.adam(lr=1e-3)

model.compile(loss = 'mean_squared_error',
              optimizer=adam)

history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              shuffle=True)
			  
model.save('dropout_layer_17_model1.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

print(history.history['loss'])
print(history.history['val_loss'])
# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('loss_plot.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('dropout_loss_17_plot.png')

