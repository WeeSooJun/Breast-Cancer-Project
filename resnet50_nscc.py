from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input, Add, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model, load_model
from keras import regularizers, optimizers
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
import os
import h5py
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from keras.callbacks import CSVLogger
from skimage import exposure
import skimage
import random


train_csv = pd.read_csv("dataset/Calc_Train_Dataset/calc_case_train.csv")
test_csv = pd.read_csv("dataset/Calc-Testset/calc_case_test.csv")

train_dir = os.path.join(os.getcwd(), 'dataset/Calc_Train_Dataset/Calc-Train/')
test_dir = os.path.join(os.getcwd(), 'dataset/Calc-Testset/Calc-Test/')


train_Y = []
for row in range(0,len(train_csv)):
    if train_csv.iloc[row]["pathology"] == "MALIGNANT":
        train_Y.append([0,1])
        train_Y.append([0,1])
        train_Y.append([0,1])
        train_Y.append([0,1])
        train_Y.append([0,1])
        train_Y.append([0,1])
    else:
        train_Y.append([1,0])
        train_Y.append([1,0])
        train_Y.append([1,0])
        train_Y.append([1,0])
        train_Y.append([1,0])
        train_Y.append([1,0])


test_Y = []
for row in range(0,len(test_csv)):
    if test_csv.iloc[row]["pathology"] == "MALIGNANT":
        test_Y.append([0,1])
    else:
        test_Y.append([1,0])

train_X_gray = []
for row in range(0,len(train_csv)):
    picture = exposure.equalize_hist(io.imread(train_dir+(train_csv.iloc[row]["image file path"]).replace("\\","/")))
    picture = color.gray2rgb(picture)
    train_X_gray.append(picture)
    train_X_gray.append(np.fliplr(picture)) #insert horizontal flip
    train_X_gray.append(np.flipud(picture))#insert vertical flip
    #train_X_gray.append(skimage.transform.rotate(picture, random.randint(45,90), resize = False,mode= "symmetric")) #rotate between 45 and 90
    #train_X_gray.append(skimage.transform.rotate(picture, random.randint(135,180), resize = False,mode= "symmetric")) #rotate between 135 and 180
    train_X_gray.append(np.rot90(picture,1))
    train_X_gray.append(np.rot90(picture,2))
    train_X_gray.append(np.rot90(picture,3))


test_X_gray = []
for row in range(0,len(test_csv)):
    picture = exposure.equalize_hist(io.imread(test_dir+(test_csv.iloc[row]["image file path"]).replace("\\","/"))) #Equalise the hist
    picture = color.gray2rgb(picture)
    test_X_gray.append(picture)

#convert to np array
train_X_gray = np.array(train_X_gray)
test_X_gray = np.array(test_X_gray)

#USING KERAS RESNET50
base_model = keras.applications.resnet50.ResNet50(include_top=False, 
                                         weights=None, 
                                         input_tensor=None, 
                                         input_shape=(256,256,3),
                                         classes=2)
base_model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

#freeze all layers
for layer in base_model.layers[:-1]:
    layer.trainable = False

calc_model_X = base_model.output
calc_model_X = GlobalAveragePooling2D()(calc_model_X)
calc_model_X = Dense(256, activation="relu", kernel_regularizer = regularizers.l2(0.01),name="fc-1")(calc_model_X)
calc_model_X = Dropout(0.5)(calc_model_X)
calc_model_X = Dense(128, activation="relu", kernel_regularizer = regularizers.l2(0.01), name="fc-2")(calc_model_X)
calc_model_X = Dropout(0.5)(calc_model_X)
calc_model_X = Dense(64, activation="relu", kernel_regularizer = regularizers.l2(0.01), name="fc-3")(calc_model_X)
calc_model_X = Dropout(0.5)(calc_model_X)
predictions = Dense(2,activation="softmax")(calc_model_X)
calc_model = Model(inputs = base_model.input, outputs=predictions)

calc_model.summary()
calc_metrics = pd.DataFrame(columns = ["loss", "acc", "val_loss", "val_acc"]) #dataframe to store



########## TRAINING
#Stage 1
calc_model.compile(Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger("metric_calc_5.csv", separator = ",", append = True)
model_calc_history = calc_model.fit(train_X_gray, train_Y, batch_size = 32,
									epochs=3, verbose = 2, validation_data = (test_X_gray,test_Y),
									callbacks = [csv_logger])

calc_metrics = calc_metrics.append(pd.DataFrame(model_calc_history.history), ignore_index=True)

#Stage 2
for layer in base_model.layers[:8]:
    layer.trainable = True
calc_model.summary()
calc_model.compile(Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_calc_history = calc_model.fit(train_X_gray, train_Y, batch_size = 32,
									epochs=10, verbose = 2, validation_data = (test_X_gray,test_Y),
									callbacks = [csv_logger])
calc_metrics = calc_metrics.append(pd.DataFrame(model_calc_history.history), ignore_index=True)

#Stage 3
for layer in base_model.layers[:]:
    layer.trainable = True
calc_model.summary()
calc_model.compile(Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_calc_history = calc_model.fit(train_X_gray, train_Y, batch_size = 32,
									epochs=37, verbose = 2, validation_data = (test_X_gray,test_Y),
									callbacks = [csv_logger])
calc_metrics = calc_metrics.append(pd.DataFrame(model_calc_history.history), ignore_index=True)
######### END OF TRAINING

#Saving Model
calc_model.save("nscc_Calc_model_6.h5")

calc_metrics = pd.DataFrame(columns = ["loss", "acc", "val_loss", "val_acc"])
calc_metrics = calc_metrics.append(pd.DataFrame(model_calc_history.history), ignore_index=True)
calc_metrics.to_csv(r"calc_metrics_6_pd_to_csv.csv")
