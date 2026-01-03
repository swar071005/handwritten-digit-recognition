import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/digit_model.h5")

img = cv2.imread("data/sample_digits/5.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0
img = img.reshape(1, 28, 28)

prediction = model.predict(img)
print("Predicted Digit:", np.argmax(prediction))
