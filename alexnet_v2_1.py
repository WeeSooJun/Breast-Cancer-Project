import skimage.io
import skimage.transform
import skimage
import tensorflow as tf
import h5py
import os
import csv
import numpy as np
import matplotlib
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger
from keras.models import load_model

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

test_data = read_csv("calc_case_test.csv")[1:]

test_x = []
test_y = []

for row in test_data:
    if row[-4] == "BENIGN" or row[-4] == "BENIGN_WITHOUT_CALLBACK":
        test_y.append([1,0])
    else:
        test_y.append([0,1])
    file_path = os.path.join('Calc-Test', row[-2].replace("\\", "/"))
    img_array = skimage.transform.resize(skimage.io.imread(file_path), (227,227,3))
    img_array = skimage.exposure.equalize_hist(img_array)
    img_array = skimage.color.gray2rgb(img_array)
    test_x.append(img_array)

test_x = np.array(test_x)
test_y = np.array(test_y)

model = model_load('alexnet_v2_calc.h5')
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

csv_logger = CSVLogger("metric_calc_6.csv", separator = ",", append = True)
model_history = model.fit(x = input_x, y = output_y, epochs=150, validation_data = (test_x, test_y), verbose = 2, callbacks = [csv_logger])
model.save("alexnet_v2_calc_1.h5")

calc_metrics = pd.DataFrame(columns = ["loss", "acc", "val_loss", "val_acc"])
calc_metrics = calc_metrics.append(pd.DataFrame(model_history.history), ignore_index=True)
calc_metrics.to_csv(r"calc_metrics_7_pd_to_csv.csv")
