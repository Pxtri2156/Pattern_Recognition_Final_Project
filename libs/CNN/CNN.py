import os 
import librosa                                                                                                                                                                                                                                                                                                                                                             
import argparse
import numpy as np 
import sys
import joblib
import json
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json


import warnings
sys.path.append("./")
from utils.loading import load_features_labels
from utils.config_parse import get_config
from utils.write_log import write_config_args_2_log
from utils.setup_log import setup_logging
import logging



class CNN:

    def __init__(self):
        self.model = None
        self.define_model()

    def define_model(self):

        # Define model
        model = Sequential()
        model.add(Conv1D(256, 5,padding='same',
                        input_shape=(216,1)))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5,padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling1D(pool_size=(8)))
        model.add(Conv1D(128, 5,padding='same',))
        model.add(Activation('relu'))
        #model.add(Conv1D(128, 5,padding='same',))
        #model.add(Activation('relu'))
        #model.add(Conv1D(128, 5,padding='same',))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        model.add(Conv1D(128, 5,padding='same',))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        self.model = model

    def train(self):
        # set optimize 
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        self.model.self.model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))
        
    def predictor(self):
        pass
    
    def save_architecture_model(self):
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

    def save_weight_model(self):
        model_name = 'Emotion_Voice_Detection_Model.h5'
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def load_architecture_model(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        self.model = model 

    def load_weight_model(self):
        self.model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
