import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

batch_size = 64
model_name1 = 'ourmodel'
model_name2 = 'Cnn'
model_name3 = 'Vgg'
model_name4 = 'Inception'


def make_dirs(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


data_path = './cross validation/testset/'
allY1 = allY2 = allY3 = allY4 = []
allProb1 = allProb2 = allProb3 = allProb4 = []
test_datagen = ImageDataGenerator(rescale=1./255)


def extract_features(directory, sample_count, model):
    features = np.zeros(shape=(sample_count, 1))
    labels = np.zeros(shape=sample_count)
    generator = test_datagen.flow_from_directory(
        directory,
        target_size=(model.input.shape[2], model.input.shape[2]),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    i = 0
    for inputs_batch, labels_batch in generator:

        features_batch = model.predict(inputs_batch)
        features_batch = (features_batch > 0.5)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return generator.filenames, labels, features


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


for fold in range(5):
    fold = fold + 1
    print("Model name: ", model_name1, "  Fold number: ", fold)
    test_path = os.path.join(data_path, str(fold), 'test')
    test_path1 = os.path.join(data_path, str(fold), 'test/include/', )
    test_path2 = os.path.join(data_path, str(fold), 'test/uninclude/', )
    model_path = os.path.join(data_path, str(fold), 'results/ourmodel', 'clude_own.h5')
    model = load_model(model_path)
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(128, 128),
                                                      batch_size=batch_size, class_mode='binary', shuffle=False)
    #prob = model.predict_generator(test_generator, steps=batch_size)
    prob = model.predict(test_generator)
    allProb1 = np.append(allProb1, prob)
    #allProb = allProb + [*np.array(prob).flat]
    sample_count = len(get_imlist(test_path1)) + len(get_imlist(test_path2))
    a = extract_features(test_path, sample_count, model)
    y = [*np.array(a[1]).flat]
    allY1 = np.append(allY1, y)
    #allY = allY + y
fold = 0


for fold in range(5):
    fold = fold + 1
    print("Model name: ", model_name2, "  Fold number: ", fold)
    test_path = os.path.join(data_path, str(fold), 'test')
    test_path1 = os.path.join(data_path, str(fold), 'test/include/', )
    test_path2 = os.path.join(data_path, str(fold), 'test/uninclude/', )
    model_path = os.path.join(data_path, str(fold), 'results/Cnn', 'clude_cnnM.h5')
    model = load_model(model_path)
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(128, 128),
                                                      batch_size=batch_size, class_mode='binary', shuffle=False)
    #prob = model.predict_generator(test_generator, steps=batch_size)
    prob = model.predict(test_generator)
    allProb2 = np.append(allProb2, prob)
    #allProb = allProb + [*np.array(prob).flat]
    sample_count = len(get_imlist(test_path1)) + len(get_imlist(test_path2))
    a = extract_features(test_path, sample_count, model)
    y = [*np.array(a[1]).flat]
    allY2 = np.append(allY2, y)
    #allY = allY + y
fold = 0


for fold in range(5):
    fold = fold + 1
    print("Model name: ", model_name3, "  Fold number: ", fold)
    test_path = os.path.join(data_path, str(fold), 'test')
    test_path1 = os.path.join(data_path, str(fold), 'test/include/', )
    test_path2 = os.path.join(data_path, str(fold), 'test/uninclude/', )
    model_path = os.path.join(data_path, str(fold), 'results/Vgg', 'clude_vgg.h5')
    model = load_model(model_path)
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(128, 128),
                                                      batch_size=batch_size, class_mode='binary', shuffle=False)
    #prob = model.predict_generator(test_generator, steps=batch_size)
    prob = model.predict(test_generator)
    allProb3 = np.append(allProb3, prob)
    #allProb = allProb + [*np.array(prob).flat]
    sample_count = len(get_imlist(test_path1)) + len(get_imlist(test_path2))
    a = extract_features(test_path, sample_count, model)
    y = [*np.array(a[1]).flat]
    allY3 = np.append(allY3, y)
    #allY = allY + y
fold = 0


for fold in range(5):
    fold = fold + 1
    print("Model name: ", model_name4, "  Fold number: ", fold)
    test_path = os.path.join(data_path, str(fold), 'test')
    test_path1 = os.path.join(data_path, str(fold), 'test/include/', )
    test_path2 = os.path.join(data_path, str(fold), 'test/uninclude/', )
    model_path = os.path.join(data_path, str(fold), 'results/Inception', 'clude_inception.h5')
    model = load_model(model_path)
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(128, 128),
                                                      batch_size=batch_size, class_mode='binary', shuffle=False)
    #prob = model.predict_generator(test_generator, steps=batch_size)
    prob = model.predict(test_generator)
    allProb4 = np.append(allProb4, prob)
    #allProb = allProb + [*np.array(prob).flat]
    sample_count = len(get_imlist(test_path1)) + len(get_imlist(test_path2))
    a = extract_features(test_path, sample_count, model)
    y = [*np.array(a[1]).flat]
    allY4 = np.append(allY4, y)
    #allY = allY + y


plt.figure(0).clf()
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot(label='ROC curve')

fpr, tpr, thresh = metrics.roc_curve(allY1, allProb1)
auc = metrics.roc_auc_score(allY1, allProb1)
plt.plot(fpr, tpr, label="own, auc=" + str(auc))

fpr, tpr, thresh = metrics.roc_curve(allY2, allProb2)
auc = metrics.roc_auc_score(allY2, allProb2)
plt.plot(fpr, tpr, label="cnn, auc=" + str(auc))

fpr, tpr, thresh = metrics.roc_curve(allY3, allProb3)
auc = metrics.roc_auc_score(allY3, allProb3)
plt.plot(fpr, tpr, label="vgg, auc=" + str(auc))

fpr, tpr, thresh = metrics.roc_curve(allY4, allProb4)
auc = metrics.roc_auc_score(allY4, allProb4)
plt.plot(fpr, tpr, label="inception, auc=" + str(auc))

plt.legend(loc=0)
plt.savefig('F:/neu/研究/影像/分类/cross validation/testset/evaluation/allROC.jpg')
plt.show()
