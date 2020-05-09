from keras.models import load_model
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import re
import matplotlib.pyplot as plt

#### Training and test data preparation #####

train_data = pd.read_csv('train.tsv', sep="\t")
test_data = pd.read_csv('test.tsv', sep="\t")
# only keep columns we care about
train_data = train_data[['Phrase', 'Sentiment']]
test_data = test_data[['Phrase']]
# Manipulation to make the reviews easier to parse/ read
train_data['Phrase'] = train_data['Phrase'].apply(lambda x: x.lower())
train_data['Phrase'] = train_data['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(train_data['Phrase'].values)
X = tokenizer.texts_to_sequences(train_data['Phrase'].values)
X = pad_sequences(X)
# Encode and get our sentiment labels
labelencoder = LabelEncoder().fit_transform(train_data['Sentiment'])
y = to_categorical(labelencoder)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)


########### CNN model evaluation ##################

cnn_model = load_model('sentiment_model_CNN_v3.h5')  # loading pre-saved CNN model

history_CNN = cnn_model.fit(X_train, Y_train, batch_size=256, epochs=2, verbose=1,
                            validation_data=(X_test, Y_test))

score_CNN = cnn_model.evaluate(X_test, Y_test, verbose=0)

history_CNN = cnn_model.fit(X_train, Y_train, batch_size=256, epochs=10, verbose=1,
                            validation_data=(X_test, Y_test))


# # plotting training and test accuracy from history for CNN ####
plt.plot(history_CNN.history['accuracy'])
plt.plot(history_CNN.history['val_accuracy'])
plt.title('CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plotting training and test loss from history
plt.plot(history_CNN.history['loss'])
plt.plot(history_CNN.history['val_loss'])
plt.title('CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


######## LSTM model evaluation ##############

lstm_model = load_model('sentiment_model_LSTM_v3.h5') # loading pre-saved LSTM model

score_LSTM = lstm_model.evaluate(X_test, Y_test, verbose=0)


history_LSTM = lstm_model.fit(X_train, Y_train, batch_size=256, epochs=10, verbose=1,
                              validation_data=(X_test, Y_test))

# # plotting training and test accuracy from history
plt.plot(history_LSTM.history['accuracy'])
plt.plot(history_LSTM.history['val_accuracy'])
plt.title('LSTM model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plotting training and test loss from history
plt.plot(history_LSTM.history['loss'])
plt.plot(history_LSTM.history['val_loss'])
plt.title('LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


### Printing the performance for both the model ####
print("CNN model performance for sentiment analysis problem is below:")
print("%s: %.2f%%" % (cnn_model.metrics_names[1], score_CNN[1] * 100))
print("%s: %.2f" % (cnn_model.metrics_names[0], score_CNN[0]))
print("\n\n")

print("LSTM model performance for sentiment analysis problem is below:")
print("%s: %.2f%%" % (lstm_model.metrics_names[1], score_LSTM[1] * 100))
print("%s: %.2f" % (lstm_model.metrics_names[0], score_LSTM[0]))