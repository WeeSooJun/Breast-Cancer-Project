import skimage
import csv
import numpy as np
import os

def read_csv(csvfilename):
    rows = ()
    with open(csvfilename) as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            rows += (tuple(row), )
    return rows

data = read_csv("calc_case_test.csv")[1:]

x = []
y = []

for row in data:
    if row[-4] == "BENIGN" or row[-4] == "BENIGN_WITHOUT_CALLBACK":
        for i in range(6):
            y.append([1,0])
    else:
        for i in range(6):
            y.append([0,1])
    file_path = os.path.join('Calc-Test', row[-2].replace("\\", "/"))
    img_array = skimage.io.imread(file_path)
##    img_array = skimage.transform.resize(img_array, (227,227,3))
##    img_array = skimage.exposure.equalize_hist(img_array)
##    img_array = skimage.color.gray2rgb(img_array)
    x.append(img_array)
##    x.append(np.fliplr(img_array))
##    x.append(np.flipud(img_array))
##    x.append(np.rot90(img_array, 1))
##    x.append(np.rot90(img_array, 2))
##    x.append(np.rot90(img_array, 3))

input_x = np.array(x)
output_y = np.array(y)

print(np.amax(input_x))
