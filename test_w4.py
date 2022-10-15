import tensorflow as tf
import numpy as np
import csv


with open('W4_Data/sign_mnist_test.csv') as file:
    csv_reader = csv.reader(file, delimiter=",", )
    next(csv_reader, None)
    all_rows = np.array([row for row in csv_reader])

    test_labels = np.array([row[0] for row in all_rows], dtype='float64')
    test_images = np.array([row[1:] for row in all_rows], dtype='float64').reshape((-1, 28, 28))

model = tf.keras.models.load_model('my_model.h5')

loss, acc = model.evaluate(test_images, test_labels)

print("Restored model, test accuracy: {:5.2f}%".format(100 * acc))