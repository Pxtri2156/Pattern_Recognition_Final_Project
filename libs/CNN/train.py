

import os 
import librosa                                                                                                                                                                                                                                                                                                                                                             
import argparse
import numpy as np 
import sys
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

import warnings
sys.path.append("./")
from utils.loading import load_features_labels
from utils.config_parse import get_config
from utils.write_log import write_config_args_2_log
from utils.setup_log import setup_logging
from libs.CNN.CNN_class import CNN
import logging


def evaluate_model(y_true, y_pred, tar_names):

    cls_report = classification_report(y_true, y_pred, target_names=tar_names )
    print("{} RESULT EVALUATION {} \n {}".format('_'*30, "_"*30, cls_report ))
    logger.info("{} RESULT EVALUATION {} \n {}".format('_'*30, "_"*30, cls_report ))


def train(cfgs):

    # Load feature data
    print("[INFO]: Step 1/8 - Loading features and labels]")
    train_features, train_labels, dev_features, dev_labels = load_features_labels(cfgs.CNN.ROOT, cfgs.CNN.DATASET_NAME)
    # print('features: ', features) 
    # print("labels: ", labels) 
    print("[INFO]: Step 2/8 - Preprocessing labels]")
    print('features shape: ', train_features.shape) 
    le = preprocessing.LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    cls_n = le.classes_
    print("class: ", cls_n) 
    # train_labels = train_labels[:24] # Must to comment
    print('len of train labels: ', len(train_labels))
    # Split data 
    print("[INFO]: Step 3/8 Split data ")
    if dev_features == []:
        print("[INFO]:  Split data with train - val : {}-{}".format( 1- cfgs.CNN.SPLIT, cfgs.CNN.SPLIT ))
        X_train , X_val, Y_train, Y_val = train_test_split(train_features, train_labels, test_size= cfgs.CNN.SPLIT, shuffle=True)
    else:
        print("[INFO]: Loading dev path")
        dev_labels = le.fit_transform(dev_labels)
        X_train, Y_train = train_features, train_labels
        X_val, Y_val = dev_features, dev_labels
    cfgs.CNN.MODEL.INPUT_SIZE = X_train.shape[1]
    print("Before fix for CNN")
    print("X train shape: ", X_train.shape)
    print('X val shape: ', X_val.shape) 
    print("Y train shape: ", len(Y_train))
    print("Y val: " , len(Y_val))

    print("[INFO]: Step 4/8 fix label and feature for CNN")
    # fix label for CNN 
    # Y_train = np_utils.to_categorical(Y_train)
    # Y_val = np_utils.to_categorical(Y_val)
    fix_Y_train = to_categorical(Y_train)
    fix_Y_val = to_categorical(Y_val)

    # Changing dimension for CNN model
    X_train =np.expand_dims(X_train, axis=2)
    X_val= np.expand_dims(X_val, axis=2)

    print("After fix for CNN")
    print("X train shape: ", X_train.shape)
    print('X val shape: ', X_val.shape) 
    print("Y train shape: ", fix_Y_train.shape)
    print("Y val: " , fix_Y_val.shape)

    # Define model
    print("[INFO]: Step 5/8 Define model CNN ")
    model = CNN(cfgs)
    # Print architecture model
    model.print_architecture_model()
    # Training
    print("[INFO]: Step 6/8 - Training ")
    model.train(X_train, X_val,  fix_Y_train, fix_Y_val)
    print("[INFO]: Completed training")
    print("[INFO]: Step 7/8 - Evaluation on validation set")
    predicted = model.predict(X_val)
    # predicted = list(predicted)
    print("predicted: ",predicted )
    evaluate_model(Y_val, predicted, cls_n)
    # print("predicted: ", le.inverse_transform(predicted))
    # Save model
    print("[INFO]: 8/8 - Saving model")
    logging.info("Model was trained")
    # joblib.dump(model, save_path)
    model.save_weight_model(cfgs.CNN.WEIGHTS_PATH)
    print("Saved model at {} ".format(cfgs.CNN.WEIGHTS_PATH))
    logging.info("Saved model at {} ".format(cfgs.CNN.WEIGHTS_PATH))


def main(cfgs):
    global logger 
    logger = setup_logging(cfgs.CNN.LOG_PATH)
    if cfgs.CNN.DISABLE_LOG == True:
      logger.disabled = False;
    write_config_to_log(cfgs, logger)
    train(cfgs)

def write_config_to_log(cfgs, logger):
    logging.info("{} CONFIGS {}".format('_'*30, "_"*30))
    logging.info("{} Split dataset train - val : {}-{}".format('\t', 1- cfgs.CNN.SPLIT, cfgs.CNN.SPLIT ))
    logging.info("{} ROOT: ".format('\t', cfgs.CNN.ROOT ))
    logging.info("{} DATASET_NAME name: {}".format('\t', cfgs.CNN.DATASET_NAME ))
    logging.info("{} FEATURES: {}".format('\t', cfgs.CNN.FEATURES ))
    logging.info("{} OUTPUT_PATH: {}".format('\t', cfgs.CNN.OUTPUT_PATH ))
    logging.info("{} WEIGHTS_PATH: {}".format('\t', cfgs.CNN.WEIGHTS_PATH ))
    logging.info("{} ARCHITECTURE_MODEL_PATH: {}".format('\t', cfgs.CNN.ARCHITECTURE_MODEL_PATH ))
    logging.info("{} VISUALIZE_LOSS_PATH: {}".format('\t', cfgs.CNN.VISUALIZE_LOSS_PATH ))
    logging.info("{} DISABLE LOG: {}".format('\t', cfgs.CNN.DISABLE_LOG))
    logging.info("{} LOG_PATH: {}".format('\t', cfgs.CNN.LOG_PATH))
    logging.info("{} MODEL: ".format('\t' ))
    logging.info("{} INPUT_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.INPUT_SIZE))
    logging.info("{} OUTPUT_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.OUTPUT_SIZE))
    logging.info("{} ACTIVATION: {}".format('\t\t', cfgs.CNN.MODEL.ACTIVATION))
    logging.info("{} PADDING: {}".format('\t\t', cfgs.CNN.MODEL.PADDING))
    logging.info("{} LEARNING_RATE: {}".format('\t\t', cfgs.CNN.MODEL.LEARNING_RATE))
    logging.info("{} DE_CAY: {}".format('\t\t', cfgs.CNN.MODEL.DE_CAY))
    logging.info("{} LOSS_FUNCTION: {}".format('\t\t', cfgs.CNN.MODEL.LOSS_FUNCTION))
    logging.info("{} METRIC: {}".format('\t\t', cfgs.CNN.MODEL.METRIC))
    logging.info("{} BATCH_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.BATCH_SIZE))
    logging.info("{} EPOCHS: {}".format('\t\t', cfgs.CNN.MODEL.EPOCHS))

    print("{}\n".format('_'*100))
def arg_parser():
    parser = argparse.ArgumentParser(description='Argument of extract data')
    parser.add_argument('--root', '-r', required=True, 
                        type=str, help='The path consists all type extracted features')
    parser.add_argument('--data_n', '-n', default='all', 
                        type=str, help='name of dataset you want to train')
    # parser.add_argument('--features', '-f', default="all", 
    #                 type=str, help='The type method that you want to extract')
    parser.add_argument('--argu', '-a', action='store_true', 
                    help='Augment data')
    parser.add_argument('--output', '-o', required=True, 
                        type=str, help='The path will save model')
    parser.add_argument('--architecture_model', '-ar', action='store_true', 
                        help='Save architecture model')
    parser.add_argument('--visualize_loss', '-v', action='store_true', 
                         help='Allow visualize loss')
    parser.add_argument('--dis_log', '-dl', action='store_true', 
                         help='Disable write log')
    parser.add_argument('--configs_file', '-c', required=True, 
                        type=str, help='The path config file for model')
    parser.add_argument(
    "--opts",
    help="Argument of Dict guided method, modify config options using the command-line 'KEY VALUE' pairs",
    default=[],
    nargs=argparse.REMAINDER,
    )

    return parser.parse_args()

def print_args(args):
    print("{} PARSER {}".format('_'*30, "_"*30))
    print("{} The path root of exacted features: \n\t{}".format('#'*3, args.root))
    print("{} Dataset name to train: \n\t{}".format('#'*3, args.data_n))
    # print("{} The methods exacted feature: \n\t{}".format('#'*3, args.features))
    print("{} Augment data: \n\t{}".format('#'*3, args.argu))
    print("{} Dataset name to train: \n\t{}".format('#'*3, args.data_n))
    print("{} The path of output : \n\t{}".format('#'*3, args.output))
    print("{} Allow save architecture model : \n\t{}".format('#'*3, args.architecture_model))
    print("{} Allow visualize loss log : \n\t{}".format('#'*3, args.visualize_loss))
    print("{} Disable write log file : \n\t{}".format('#'*3, args.dis_log))
    print("{} The path of configs file : \n\t{}".format('#'*3, args.configs_file))

def print_cfgs(cfgs):
    print("{} CONFIGS {}".format('_'*30, "_"*30))
    print("{} Split dataset train - val : {}-{}".format('\t', 1- cfgs.CNN.SPLIT, cfgs.CNN.SPLIT ))
    print("{} ROOT: ".format('\t', cfgs.CNN.ROOT ))
    print("{} DATASET_NAME name: {}".format('\t', cfgs.CNN.DATASET_NAME ))
    print("{} FEATURES: {}".format('\t', cfgs.CNN.FEATURES ))
    print("{} OUTPUT_PATH: {}".format('\t', cfgs.CNN.OUTPUT_PATH ))
    print("{} WEIGHTS_PATH: {}".format('\t', cfgs.CNN.WEIGHTS_PATH ))
    print("{} ARCHITECTURE_MODEL_PATH: {}".format('\t', cfgs.CNN.ARCHITECTURE_MODEL_PATH ))
    print("{} VISUALIZE_LOSS_PATH: {}".format('\t', cfgs.CNN.VISUALIZE_LOSS_PATH ))
    print("{} LOG_PATH: {}".format('\t', cfgs.CNN.LOG ))
    print("{} MODEL: ".format('\t' ))
    print("{} INPUT_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.INPUT_SIZE))
    print("{} OUTPUT_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.OUTPUT_SIZE))
    print("{} ACTIVATION: {}".format('\t\t', cfgs.CNN.MODEL.ACTIVATION))
    print("{} PADDING: {}".format('\t\t', cfgs.CNN.MODEL.PADDING))
    print("{} LEARNING_RATE: {}".format('\t\t', cfgs.CNN.MODEL.LEARNING_RATE))
    print("{} DE_CAY: {}".format('\t\t', cfgs.CNN.MODEL.DE_CAY))
    print("{} LOSS_FUNCTION: {}".format('\t\t', cfgs.CNN.MODEL.LOSS_FUNCTION))
    print("{} METRIC: {}".format('\t\t', cfgs.CNN.MODEL.METRIC))
    print("{} BATCH_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.BATCH_SIZE))
    print("{} EPOCHS: {}".format('\t\t', cfgs.CNN.MODEL.EPOCHS))

    print("{}\n".format('_'*100))

def setup_config(cfgs, args):
    cfgs.CNN.ROOT = args.root
    cfgs.CNN.DATASET_NAME  = args.data_n
    cfgs.CNN.ARGUMENT = args.argu
    cfgs.CNN.OUTPUT_PATH = args.output
    cfgs.CNN.WEIGHTS_PATH = os.path.join(args.output, "model.h5")
    if args.architecture_model == True:
        cfgs.CNN.ARCHITECTURE_MODEL_PATH = os.path.join(args.output, "architecture_model.json")
    if args.visualize_loss == True:
        cfgs.CNN.VISUALIZE_LOSS_PATH =  os.path.join(args.output, "loss.png")
    cfgs.CNN.LOG = args.dis_log
    cfgs.CNN.LOG_PATH =  os.path.join(args.output, "log.log")

    return cfgs
if __name__ == "__main__":
    args = arg_parser()
    cfgs = get_config()
    cfgs.merge_from_file('./configs/CNN/CNN.yaml')
    if not os.path.isfile(args.configs_file):
        print("File configs not exits!!")
    # old_stdout = sys.stdout
    # log_file = open(os.path.join(args.output, 'log.log'),"w")
    # sys.stdout = log_file
    cfgs.merge_from_file(args.configs_file)
    print_args(args)
    cfgs = setup_config(cfgs, args)
    print_cfgs(cfgs)
    main(cfgs)
    # sys.stdout = old_stdout
    # log_file.close()

'''
    -r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\feature_data \
    -n TESS \
    -o E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\models\CNN  \
    -c E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\CNN\CNN.yaml

-r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\feature_data -n TESS -o E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\models\CNN  -c E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\CNN\CNN.yaml

'''

