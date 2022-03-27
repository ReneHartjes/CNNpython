import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ["Dog","Cat","Wild"]


def prepare(filepath):
    IMG_SIZE = 70
    img_array = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    new_array= cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))


    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)




model = tf.keras.models.load_model('64x3-CNN.model')

prediction = model.predict([prepare('dotes.jpg')])
print(prediction)