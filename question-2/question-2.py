"""
2. Implement *Logistic Regression on following dataset https://www.kaggle.com/ronitf/heart-disease-uci .
    a. Normalize the data before feeding it to the model
    b. Show the Loss on TensorBoard
    c. Change three hyperparameter and report how the accuracy changes

*Logistic regression: for understanding the difference between Linear Regression and Logistic Regression refer to this link: https://stackoverflow.com/questions/12146914/what-is-the-difference-between-linear-regression-and-logistic-regression
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

out_file = open("output\\log.txt", "a")


path = "heart.csv"
data_df = pd.read_csv(path)

print(data_df)

# drop nulls
data_df = data_df.dropna()

# normalize
scaler = MinMaxScaler()
scaler.fit(data_df)
normalized_data = scaler.transform(data_df)
print(normalized_data)

# encode
x = normalized_data[:, :-1]
y = normalized_data[:, -1]


# split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)

in_shape = x_train.shape
out_shape = (None, 1)
avg_neuron = int((in_shape[1] + out_shape[1]) / 2)

# base model
model_name = "base"
out_file.write(model_name + ":\n")

model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, initial_epoch=0)
model.summary()


loss, accuracy = model.evaluate(x_test, y_test)

out_file.write("""
ORIGINAL LOSS: {loss}
ORIGINAL ACCURACY: {accuracy}
""".format(loss=loss, accuracy=accuracy))

out_file.write("\n\n")


# more epochs
model_name = "more_epochs"
out_file.write(model_name + ":\n")

model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, initial_epoch=0)
model.summary()

loss2, accuracy2 = model.evaluate(x_test, y_test)


out_file.write("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL ACCURACY: {accuracy}
CHANGE: {accuracy_change}
NEW ACCURACY: {accuracy2}
""".format(loss=loss, loss2=loss2, loss_change=loss2-loss, accuracy=accuracy, accuracy_change= accuracy2 - accuracy, accuracy2=accuracy2))

out_file.write("\n\n")

# more hidden layers
model_name = "more_hidden_layers"
out_file.write(model_name + ":\n")

model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, initial_epoch=0)
model.summary()

loss2, accuracy2 = model.evaluate(x_test, y_test)


out_file.write("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL ACCURACY: {accuracy}
CHANGE: {accuracy_change}
NEW ACCURACY: {accuracy2}
""".format(loss=loss, loss2=loss2, loss_change=loss2-loss, accuracy=accuracy, accuracy_change= accuracy2 - accuracy, accuracy2=accuracy2))

out_file.write("\n\n")

# sigmoid activation
model_name = "sigmoid_activation"
out_file.write(model_name + ":\n")

model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='sigmoid'))
model.add(Dense(avg_neuron, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, initial_epoch=0)
model.summary()

loss2, accuracy2 = model.evaluate(x_test, y_test)


out_file.write("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL ACCURACY: {accuracy}
CHANGE: {accuracy_change}
NEW ACCURACY: {accuracy2}
""".format(loss=loss, loss2=loss2, loss_change=loss2-loss, accuracy=accuracy, accuracy_change= accuracy2 - accuracy, accuracy2=accuracy2))

out_file.write("\n\n")


out_file.close()