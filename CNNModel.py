from keras import models, layers

class CNNModel:
    @staticmethod
    def generateModel(input):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input.shape[1:]))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model
    @staticmethod
    def createsentimentLTSMmodel(max_features):
        model = models.Sequential()
        model.add(layers.Embedding(max_features, 128,input_length = 45))
        model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.Dense(5,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        return model  
    @staticmethod
    def createsentimentCNNmodel(max_features):
        model = models.Sequential()
        model.add(layers.Embedding(max_features, 128,input_length = 45))
        model.add(layers.Flatten())
        model.add(layers.Dense(32,activation='relu'))    
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(64,activation='relu'))    
        model.add(layers.Dropout(rate=0.33))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(5,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        return model  