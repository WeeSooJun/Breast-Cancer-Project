from keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from skimage import exposure
import matplotlib.pyplot as plt
import csv
import os
import cv2
import warnings
import numpy as np


csvFilename = 'calc_case_test.csv'
modelName = 'alexnet_calc_9'
test_dir = os.path.join(os.getcwd(), 'Calc-Test')

with open(csvFilename, 'r') as f:
    labels = list(csv.reader(f, delimiter = ','))
x = []
y = []
for label in labels[1:]:
    im = cv2.imread(os.path.join(test_dir,label[-2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_eq = exposure.equalize_adapthist(im)        
    res = cv2.resize(img_eq, dsize=(227,227), interpolation = cv2.INTER_CUBIC)
    x.append(res)
    if ((label[9] == 'BENIGN') | (label[9] == 'BENIGN_WITHOUT_CALLBACK')):
        y.append([1,0]) # we use one-hot encoding here
    elif (label[9] == 'MALIGNANT'):
        y.append([0,1])
x = np.array(x)
y = np.array(y)


model = load_model(modelName + ".h5")

y_pred = model.predict(x)
y_pred = y_pred.ravel() #flatten both pred and true y
y = y.ravel()


fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='AlexNet (area = {:.3f})'.format(auc_keras))
plt.legend(loc='best')
plt.title('ROC curve for AlexNet')
plt.show()
