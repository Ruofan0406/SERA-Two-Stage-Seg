import os
import numpy as np
import time
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import layers
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import SGD
from keras import regularizers
from keras.models import load_model
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

train_path = './Data-clude/train/'
validation_path = './Data-clude/validation/'
test_path = './Data-clude/test/'
target_path = './Ownresult/'


train_num = 739
validation_num = 244
test_num = 254
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
weight_decay = 1e-6


train_generator = train_datagen.flow_from_directory(train_path, target_size=(128, 128), batch_size=64,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_path, target_size=(128, 128), batch_size=20,
                                                        class_mode='binary')


model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(128, 128, 3),
                 kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten()) ##将输入层的数据压成一维的数据
model.add(layers.Dense(120, activation='relu')) #全连接
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4, decay=1e-6), metrics=['acc'])
model.summary()

training_start = np.int64(time.strftime('%H%M%S', time.localtime(time.time())))
## %H 24小时制小时数（0-23） %M 分钟数（0-59） %S 秒（00-59）
model_checkpoint = ModelCheckpoint(target_path + '/clude_own.h5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(train_generator, steps_per_epoch=20, epochs=1000,
                              validation_data=validation_generator, validation_steps=20,
                              callbacks=[model_checkpoint]) ##分批次的读取数据，节省内存
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
training_end = np.int64(time.strftime('%H%M%S', time.localtime(time.time())))
print(training_end - training_start)


plt.plot(epochs, acc, 'r', label='Train acc')
plt.plot(epochs, loss, 'g', label='Train loss')
plt.plot(epochs, val_acc, 'b', label='Val acc')
plt.plot(epochs, val_loss, 'k', label='Val loss')
# plt.grid(True)
plt.xlabel("Epochs")
plt.ylabel('binary_crossentropy loss and accuracy')
plt.legend(loc="upper right")
plt.title('Training and validation loss/acc:own')
plt.savefig(target_path + '/Training and validation loss.png')
