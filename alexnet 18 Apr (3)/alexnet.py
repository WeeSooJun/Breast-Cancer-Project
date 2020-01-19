import skimage
import tensorflow as tf
import h5py
import os
import csv
import numpy as np
import matplotlib
import pandas as pd
from keras import layers
from keras.models import Model
from keras.callbacks import CSVLogger

base_dir = os.path.join(os.getcwd())
train_dir = os.path.join(base_dir, 'train')
train_dir_benign = os.path.join(train_dir, "BENIGN")
train_dir_malignant = os.path.join(train_dir, "MALIGNANT")

def read_csv(csvfilename):
    rows = ()
    with open(csvfilename) as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            rows += (tuple(row), )
    return rows

def alexnet(in_shape=(227,227,3), n_classes=2, opt='adam'):
    in_layer = layers.Input(in_shape)
    avg_pool1 = layers.AveragePooling2D()(in_layer)
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(avg_pool1)
    pool1 = layers.AveragePooling2D()(conv1)
    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)
    pool2 = layers.AveragePooling2D()(conv2)
    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)
    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)
    pool3 = layers.AveragePooling2D()(conv4)
    flattened = layers.Flatten()(pool3)
    dense1 = layers.Dense(4096, activation='relu')(flattened)
    drop1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(4096, activation='relu')(drop1)
    drop2 = layers.Dropout(0.5)(dense2)
    preds = layers.Dense(n_classes, activation='softmax')(drop2)

    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
	              metrics=["accuracy"])
    return model

data = read_csv("calc_case_train.csv")[1:]

x = []
y = []

for row in data:
    if row[-4] == "BENIGN" or row[-4] == "BENIGN_WITHOUT_CALLBACK":
        for i in range(6):
            y.append([1,0])
    else:
        for i in range(6):
            y.append([0,1])
    file_path = os.path.join('Calc-Train', row[-2].replace("\\", "/"))
    img_array = skimage.transform.resize(skimage.io.imread(file_path), (227,227,3))
    img_array = skimage.exposure.equalize_hist(img_array)
    img_array = skimage.color.gray2rgb(img_array)
    x.append(img_array)
    x.append(np.fliplr(img_array))
    x.append(np.flipud(img_array))
    x.append(np.rot90(img_array, 1))
    x.append(np.rot90(img_array, 2))
    x.append(np.rot90(img_array, 3))

input_x = np.array(x)
output_y = np.array(y)

model = alexnet()
model_history = model.fit(x = input_x, y = output_y, epochs=50, validation_split=0.1, verbose = 2, callbacks = [csv_logger])
model.save("alexnet_v3_calc.h5")

calc_metrics = pd.DataFrame(columns = ["loss", "acc", "val_loss", "val_acc"])
calc_metrics = calc_metrics.append(pd.DataFrame(model_history.history), ignore_index=True)
calc_metrics.to_csv(r"calc_metrics_6_pd_to_csv.csv")

##y_pred = model.predict(input_x)
##y_pred = y_pred.ravel() #flatten both pred and true y
##y = y.ravel()
##
##
##fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred)
##auc_keras = auc(fpr_keras, tpr_keras)
##
##plt.figure(1)
##plt.plot([0, 1], [0, 1], 'k--')
##plt.plot(fpr_keras, tpr_keras, label='AlexNet (area = {:.3f})'.format(auc_keras))
##plt.legend(loc='best')
##plt.title('ROC curve for AlexNet')
##plt.show()

##train_datagen = ImageDataGenerator() #samplewise_center = True, samplewise_std_normalization = True, featurewise_center = True, featurewise_std_normalization = True)
##train_generator = train_datagen.flow_from_directory(
##        train_dir,
##        target_size=(256, 256),
##        color_mode="rgb",
##        batch_size=20,
##        class_mode='categorical',
##        subset = 'training',
##        save_to_dir=base_dir+"\\augmented")
##
##b = 10
##for batch in train_generator:
##    if b==0:
##        break
##    else:
##        b-=1

##validation_generator = train_datagen.flow_from_directory(
##        train_dir,
##        target_size=(227, 227),
##        batch_size=20,
##        class_mode='categorical',
##        subset = 'validation')

##model = alexnet()
##print(model.summary())
##model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data = validation_generator, validation_steps = 50)
##model.save("alexnet_v2.h5")
