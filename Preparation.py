import pandas as pd
import os
import csv
import cv2
import shutil

def read_csv(csvfilename):
    rows = ()
    with open(csvfilename) as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            rows += (tuple(row), )
    return rows

#Total = 1317 data
#Train_index = 659
#validation_index = 659

base_dir = os.path.join(os.getcwd())
train_dir = os.path.join(base_dir, 'train')
train_dir_benign = os.path.join(train_dir, "BENIGN")
train_dir_malignant = os.path.join(train_dir, "MALIGNANT")

test_dir = os.path.join(base_dir, 'test')
test_dir_benign = os.path.join(test_dir, "BENIGN")
test_dir_malignant = os.path.join(test_dir, "MALIGNANT")


data = read_csv('mass_case_test.csv')

m = 1
n = 1

for row in data[1:]:
    pathology = row[-4]
    file_path = os.path.join('Mass-Test', row[-2])
    base = os.path.basename(file_path)
    if pathology == "BENIGN" or pathology == "BENIGN_WITHOUT_CALLBACK":
        dst = os.path.join(test_dir_benign, str(m) + base)
        m += 1
        shutil.copy(file_path, dst)
    else:
        dst = os.path.join(test_dir_malignant, str(n) + base)
        n += 1
        shutil.copy(file_path, dst)
