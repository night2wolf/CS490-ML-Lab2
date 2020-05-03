from CNNModel import CNNModel
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# Gets all our labels by directory names
labels = os.listdir('./natural-images/data/natural_images/')
print(labels)
# Retrieve all our images as a numpy array
x_data =[]
y_data = []
for label in labels:
    path = './natural-images/data/natural_images/{0}/'.format(label)
    folder_data = os.listdir(path)
    for image_path in folder_data:
        image = cv2.imread(path+image_path)
        image_resized = cv2.resize(image, (32,32))
        x_data.append(np.array(image_resized))
        y_data.append(label)

x_data = np.array(x_data)
y_data = np.array(y_data)
print(x_data.shape)
print(y_data.shape)
# Prepare our Label set
y_encoded = LabelEncoder().fit_transform(y_data)
y_categories = to_categorical(y_encoded)

# Split up our data for training
X_train, X_test, Y_train, Y_test = train_test_split(x_data,y_categories, test_size = 0.2)
model = CNNModel.generateModel(x_data)
history = model.fit(X_train,Y_train, epochs=25, validation_split=0.2)
print(history.history.keys())
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()