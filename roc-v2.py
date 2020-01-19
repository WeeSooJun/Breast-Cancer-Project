from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input, Add, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers, optimizers
from numpy.testing import assert_allclose
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
import h5py
import pandas as pd
import numpy as np
from numpy import ravel
import csv
import matplotlib.pyplot as plt
from skimage import transform
import skimage
import skimage.io
from skimage import io
from skimage import exposure
from skimage import color
from sklearn.metrics import auc, roc_curve

base_dir = os.path.join(os.getcwd())
train_dir = os.path.join(base_dir, 'train')


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


alexnet = load_model('alexnet_v2_calc.h5')

y_pred = alexnet.predict(test_x).ravel()
y = ravel(test_y)
fpr, tpr, thresholds = roc_curve(y, y_pred)
auc_resnet = auc(fpr,tpr)
# plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='VGG-V1 (area = {:.3f})'.format(auc_resnet), color='red')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
#plt.show()
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
#plt.show()
# y_pred = resnetV3.predict(test_X_gray).ravel()
# y = ravel(test_Y)
# fpr, tpr, thresholds = roc_curve(y, y_pred)
# auc_resnet = auc(fpr,tpr)
# plt.figure(3)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='ResNet-V3 (area = {:.3f})'.format(auc_resnet), color='blue')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
