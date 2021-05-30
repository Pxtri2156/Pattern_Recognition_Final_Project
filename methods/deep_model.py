import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
class CNN():
    def __init__():
        pass

    def train(X, Y):
        pass

    def inference(X):
        pass

class example_deep():

    def init(self):
        model = None

    def build(self):
        this.model=Sequential()
        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        model.add(Dropout(0.2))

        model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=8, activation='softmax'))
        model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
        model.summary()

    def fit(self, x_train, x_test, y_train , y_test):
        rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
        history= this.model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

    def predict(self, x_test):
        return this.model.predict(x_test)
