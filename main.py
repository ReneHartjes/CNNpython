import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


DATADIR =(r"./train")
CATEGORIES = ["Dog","Cat","Wild"]

for category in CATEGORIES:
    path= os.path.join(DATADIR, category) #path to cats or dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
        #print(img_array)################LOAD THE IMAGES FROM DIR
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

IMG_SIZE = 70
#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


training_data = []

def create_training_data():
    for category in CATEGORIES:
        path= os.path.join(DATADIR, category) #path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                IMG_SIZE = 70
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, class_num])

            except Exception as e:
                pass
create_training_data()

print(len(training_data))
print("done")
print(training_data)


import random
random.shuffle(training_data)
for sample in training_data:
    print(sample[1])
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y) #.reshape(-1, IMG_SIZE, IMG_SIZE, 1) ####1 is greyscale 3 ist color

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

'''
zum Reinladen:

pickle_in = open("X.pickle", "rb")
X=pickle.load(pickle_in)
X[1]

'''