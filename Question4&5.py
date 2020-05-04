import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import re
from CNNModel import CNNModel
train_data = pd.read_csv('./sentiment-analysis-on-movie-reviews/train.tsv',sep="\t")
test_data = pd.read_csv('./sentiment-analysis-on-movie-reviews/test.tsv',sep="\t")
# only keep columns we care about
train_data = train_data[['Phrase','Sentiment']]
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
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
# Train using a CNN Model (Question 4)
model = CNNModel.createsentimentCNNmodel(max_features)
history = model.fit(X_train,Y_train,batch_size=32,epochs=7)
score_cnn,acc_cnn = model.evaluate(X_test,Y_test,verbose=2,batch_size=32)
# Train using a LTSM Model (Question 5)
model_ltsm = CNNModel.createsentimentLTSMmodel(max_features)
history = model_ltsm.fit(X_train,Y_train,batch_size=32,epochs=7)
score_ltsm,acc_ltsm = model_ltsm.evaluate(X_test,Y_test,verbose=2,batch_size=32)
print("Score of CNN Model %.2f"%(score_cnn))
print("Accuracy of CNN Model %.2f"%(acc_cnn))
print("Score of TLSM Model %.2f"%(score_ltsm))
print("Accuracy of LTSM Model %.2f"%(acc_ltsm))
