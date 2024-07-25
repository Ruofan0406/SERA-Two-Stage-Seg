import os
import numpy as np
import time
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Activation, Flatten
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from model.ourmodel import ourModel

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def make_dirs(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


# modify the name
model_name = 'ourmodel'

for fold in range(5):
    fold += 1
    data_path = './clude_fold/Datasets/'
    print("Model name: ", model_name, "  Fold number: ", fold)

    train_path = os.path.join(data_path, str(fold), 'train')
    # validation_path = os.path.join(data_path, str(fold), 'validation')
    test_path = os.path.join(data_path, str(fold), 'test')

    target_path = os.path.join(data_path, str(fold), 'results', model_name)
    make_dirs(target_path)

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_path, target_size=(128, 128), batch_size=64,
                                                        class_mode='binary')
    # validation_generator = test_datagen.flow_from_directory(validation_path, target_size=(128, 128), batch_size=20,
    #                                                         class_mode='binary')
    model = ourModel()
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4, decay=1e-6), metrics=['acc'])
    model.summary()
    training_start = np.int64(time.strftime('%H%M%S', time.localtime(time.time())))
    model_checkpoint = ModelCheckpoint(target_path + '/clude_own.h5', monitor='loss', verbose=1, save_best_only=True)
    history = model.fit_generator(train_generator, steps_per_epoch=20, epochs=1000,
                                  callbacks=[model_checkpoint])
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(acc) + 1)
    # training_end = np.int64(time.strftime('%H%M%S', time.localtime(time.time())))
    # print(training_end - training_start)
    #
    #
    # plt.plot(epochs, acc, 'r', label='Train acc')
    # plt.plot(epochs, loss, 'g', label='Train loss')
    # plt.plot(epochs, val_acc, 'b', label='Val acc')
    # plt.plot(epochs, val_loss, 'k', label='Val loss')
    # # plt.grid(True)
    # plt.xlabel("Epochs")
    # plt.ylabel('binary_crossentropy loss and accuracy')
    # plt.legend(loc="upper right")
    # plt.title('Training and validation loss/acc:own')
    # plt.savefig(target_path + '/Training and validation loss.png')
