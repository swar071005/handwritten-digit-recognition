import tensorflow as tf

model = tf.keras.models.load_model("model/digit_model.h5")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test / 255.0

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
