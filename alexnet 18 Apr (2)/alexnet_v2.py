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


np.random.seed(1000)
#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(227*227*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

csv_logger = CSVLogger("metric_calc_5.csv", separator = ",", append = True)
model_history = model.fit(x = input_x, y = output_y, epochs=50, validation_data = (test_x, test_y), verbose = 2, callbacks = [csv_logger])
model.save("alexnet_v2_calc.h5")

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
