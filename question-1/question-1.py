"""
1. Build a Sequential model using keras to implement Linear Regression with any data set of your choice except the datasets being discussed in the class or used before
    a. Show the graph on TensorBoard
    b. Plot the loss and then change the below parameter and report your view how the result changes in each case
        a.	learning rate
        b.	batch size
        c.	optimizer
        d.	activation function
"""

# Thanks to: https://www.tensorflow.org/tutorials/keras/regression

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import get_file
from keras.optimizers import RMSprop, SGD

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_df_with_targets_at_the_end(df, targets: list):
    return df[[col for col in df if col not in targets] + targets]


out_file = open("output\\log.txt", "a")

dataset_path = get_file("auto-mpg.data",
                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# drop nulls
dataset = dataset.dropna()

# numeric to one-hot-categorical
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()

dataset = get_df_with_targets_at_the_end(dataset, ["MPG"])
dataset = dataset.values

# split
x_train, x_test, y_train, y_test = train_test_split(dataset[:, :-1], dataset[:, -1], test_size=0.2, random_state=87)

print("X:")
print(x_train)

print("Y:")
print(y_train)

in_shape = x_train.shape
out_shape = (None, 1)
avg_neuron = int((in_shape[1] + out_shape[1]) / 2)

# base
model_name = "base"
out_file.write(model_name + ":\n")
model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))

# model.add(Dense(64, input_dim=in_shape[1], activation='relu'))
# model.add(Dense(64, activation='relu'))

model.add(Dense(1))
model.compile(loss='mse', optimizer=RMSprop(.001), metrics=['mae', 'mse'])

history = model.fit(x_train, y_train, epochs=100, initial_epoch=0)
model.summary()

loss, mae, mse = model.evaluate(x_test, y_test)
out_file.write("LOSS: {}\n".format(loss))
out_file.write("MEAN SQUARED ERROR: {}\n".format(mse))
out_file.write("MEAN ABSOLUTE ERROR: {}\n".format(mae))
out_file.write("\n\n")

for key in history.history:
    plt.plot(history.history[key])
    plt.title("{} vs Epoch".format(key))
    plt.ylabel(key)
    plt.xlabel('Epoch')
    file_name = model_name + "_" + key + ".png"
    plt.savefig("output\\" + file_name)
    # plt.show()

# increase learning rate
model_name = "increased_learning_rate"
out_file.write(model_name + ":\n")
model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))

model.add(Dense(1))
model.compile(loss='mse', optimizer=RMSprop(.01), metrics=['mae', 'mse'])

history = model.fit(x_train, y_train, epochs=100, initial_epoch=0)
model.summary()

loss2, mae2, mse2 = model.evaluate(x_test, y_test)

out_file.write("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL MEAN SQUARED ERROR: {mse}
CHANGE: {mse_change}
NEW MEAN SQUARED ERROR: {mse2}

ORIGINAL MEAN SQUARED ERROR: {mae}
CHANGE: {mae_change}
NEW MEAN SQUARED ERROR: {mae2}
""".format(loss=loss, loss2=loss2, loss_change=loss2 - loss, mse=mse, mse_change=mse2 - mse, mse2=mse2, mae=mae,
           mae_change=mae2 - mae, mae2=mae2))

out_file.write("\n\n")

for key in history.history:
    plt.plot(history.history[key])
    plt.title("{} vs Epoch".format(key))
    plt.ylabel(key)
    plt.xlabel('Epoch')
    file_name = model_name + "_" + key + ".png"
    plt.savefig("output\\" + file_name)
    # plt.show()

# increase batch size
model_name = "increased_batch_size"
out_file.write(model_name + ":\n")
model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))

model.add(Dense(1))
model.compile(loss='mse', optimizer=RMSprop(.001), metrics=['mae', 'mse'])

history = model.fit(x_train, y_train, epochs=100, initial_epoch=0, batch_size=64)
model.summary()

loss2, mae2, mse2 = model.evaluate(x_test, y_test)
out_file.write("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL MEAN SQUARED ERROR: {mse}
CHANGE: {mse_change}
NEW MEAN SQUARED ERROR: {mse2}

ORIGINAL MEAN SQUARED ERROR: {mae}
CHANGE: {mae_change}
NEW MEAN SQUARED ERROR: {mae2}
""".format(loss=loss, loss2=loss2, loss_change=loss2 - loss, mse=mse, mse_change=mse2 - mse, mse2=mse2, mae=mae,
           mae_change=mae2 - mae, mae2=mae2))

out_file.write("\n\n")

for key in history.history:
    plt.plot(history.history[key])
    plt.title("{} vs Epoch".format(key))
    plt.ylabel(key)
    plt.xlabel('Epoch')
    file_name = model_name + "_" + key + ".png"
    plt.savefig("output\\" + file_name)
    # plt.show()

# change optimizer
model_name = "sgd_optimizer"
out_file.write(model_name + ":\n")
model = Sequential()

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='relu'))
model.add(Dense(avg_neuron, activation='relu'))

# model.add(Dense(64, input_dim=in_shape[1], activation='relu'))
# model.add(Dense(64, activation='relu'))

model.add(Dense(1))
model.compile(loss='mse', optimizer=SGD(.001), metrics=['mae', 'mse'])

history = model.fit(x_train, y_train, epochs=100, initial_epoch=0)
model.summary()

loss2, mae2, mse2 = model.evaluate(x_test, y_test)
out_file.write("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL MEAN SQUARED ERROR: {mse}
CHANGE: {mse_change}
NEW MEAN SQUARED ERROR: {mse2}

ORIGINAL MEAN SQUARED ERROR: {mae}
CHANGE: {mae_change}
NEW MEAN SQUARED ERROR: {mae2}
""".format(loss=loss, loss2=loss2, loss_change=loss2 - loss, mse=mse, mse_change=mse2 - mse, mse2=mse2, mae=mae,
           mae_change=mae2 - mae, mae2=mae2))

out_file.write("\n\n")

for key in history.history:
    plt.plot(history.history[key])
    plt.title("{} vs Epoch".format(key))
    plt.ylabel(key)
    plt.xlabel('Epoch')
    file_name = model_name + "_" + key + ".png"
    plt.savefig("output\\" + file_name)
    # plt.show()

# Change activation function
model = Sequential()
model_name = "sigmoid_activation"
out_file.write(model_name + ":\n")

# input layer has neurons of shape + 1 for bias
model.add(Dense(in_shape[1] + 1, input_dim=in_shape[1], activation='sigmoid'))
model.add(Dense(avg_neuron, activation='sigmoid'))

model.add(Dense(1))
model.compile(loss='mse', optimizer=RMSprop(.001), metrics=['mae', 'mse'])

history = model.fit(x_train, y_train, epochs=100, initial_epoch=0)
model.summary()

loss2, mae2, mse2 = model.evaluate(x_test, y_test)
out_file.write("""
ORIGINAL LOSS: {loss}
CHANGE: {loss_change}
NEW LOSS: {loss2}

ORIGINAL MEAN SQUARED ERROR: {mse}
CHANGE: {mse_change}
NEW MEAN SQUARED ERROR: {mse2}

ORIGINAL MEAN SQUARED ERROR: {mae}
CHANGE: {mae_change}
NEW MEAN SQUARED ERROR: {mae2}
""".format(loss=loss, loss2=loss2, loss_change=loss2 - loss, mse=mse, mse_change=mse2 - mse, mse2=mse2, mae=mae,
           mae_change=mae2 - mae, mae2=mae2))

out_file.write("\n\n")

for key in history.history:
    plt.plot(history.history[key])
    plt.title("{} vs Epoch".format(key))
    plt.ylabel(key)
    plt.xlabel('Epoch')
    file_name = model_name + "_" + key + ".png"
    plt.savefig("output\\" + file_name)
    # plt.show()

out_file.close()