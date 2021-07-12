import os 
import librosa                                                                                                                                                                                                                                                                                                                                                             
import argparse
import numpy as np 
import sys
import joblib
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import tensorflow.keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import model_from_json


import warnings
sys.path.append("./")
from utils.loading import load_features_labels
from utils.config_parse import get_config
from utils.write_log import write_config_args_2_log
from utils.setup_log import setup_logging
import logging

class MINI_VGG:

    def __init__(self, cfgs):
        self.log_train = None
        self.resume = False
        print("Called init function")
        self.model = None
        self.input_size = cfgs.MINI_VGG.MODEL.INPUT_SIZE
        self.output_size = cfgs.MINI_VGG.MODEL.OUTPUT_SIZE
        self.epochs = cfgs.MINI_VGG.MODEL.EPOCHS
        self.batch_size = cfgs.MINI_VGG.MODEL.BATCH_SIZE
        self.define_model()

    def define_model(self):

        # Define model
        model = Sequential()
        # inputShape = (height, width, depth)
        # chanDim = -1
        # if using "channels first", update the input shape
        # and channels dimension
        # if K.image_data_format() == "channels_first":
        #     inputShape = (depth, height, width)
        #     chanDim = 1

        # first CONV => RELU => BN => CONV 
        # => RELU => BN => POOL => DO
        model.add(Conv1D(32, 3,padding='same',
                        input_shape=(self.input_size,1)))

        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv1D(32, 3,padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=(2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => BN => CONV 
        # => RELU => BN => POOL => DO
        model.add(Conv1D(64, 3,padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv1D(64, 3,padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=(2)))
        model.add(Dropout(0.25))

        # first (only) FC => RELU => BN => DO
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(self.output_size))
        model.add(Activation("softmax"))
        self.model = model

    def print_architecture_model(self):
        self.model.summary()

    def train(self,x_traincnn, x_testcnn, y_train, y_test):
        assert(len(x_traincnn.shape) == 3 ), "x_traincnn must be tensor (dimenson = 3)"
        assert(len(x_testcnn.shape) == 3 ), "x_testcnn must be tensor (dimenson = 3)"
        
        # set optimize 
        opt = tensorflow.keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        self.log_train = self.model.fit(x_traincnn, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_testcnn, y_test))
        
    def predict(self, x_tess):
        preds = self.model.predict(x_tess, 
                         batch_size=self.batch_size, 
                         verbose=1)
        preds = preds.argmax(axis=1) 
        return preds
    
    def plot_loss(self, plot_file):
        plt.plot(self.log_train.history['loss'])
        plt.plot(self.log_train.history['val_loss'])
        plt.title('MINI_VGG model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(plot_file)
    
    def save_architecture_model(self, json_path):
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)

    def save_weight_model(self, model_path):
        model_name = 'Emotion_Voice_Detection_Model.h5'
        if model_path == "":
            save_dir = os.path.join(os.getcwd(), 'saved_models')
        else:
            save_dir = model_path
        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def load_architecture_model(self, input_path):
        '''
            input_path: json file
        '''
        assert(os.path.isfile(input_path)),"Model architecture path not exist or wrong!!!"
        json_file = open(input_path, 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        self.model = model 
        json_file.close()

    def load_weight_model(self, model_path):
        assert(os.path.isfile(model_path)),"Model weight path not exist or wrong!!!"
        self.model.load_weights(model_path)
