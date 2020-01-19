import skimage.io
import skimage.transform
import skimage
import tensorflow as tf
import h5py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

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

model = load_model("alexnet_v2_calc_18_Apr_(2).h5")
y_pred = model.predict(test_x)
y_true = test_y.argmax(1)
y_pred = y_pred.argmax(1)
print("y_true")
print(y_true)
print("y_pred")
print(y_pred)

labels = [0, 1]
cm = confusion_matrix(y_true, y_pred, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
