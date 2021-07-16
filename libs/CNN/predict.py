

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


def demo_cnn(root_path, audio_path):

    # Extract feature data



def main(cfgs):
    global logger 
    logger = setup_logging(cfgs.CNN.LOG_PATH)
    if cfgs.CNN.DISABLE_LOG == True:
      logger.disabled = False;
    write_config_to_log(cfgs, logger)
    train(cfgs)

def write_config_to_log(cfgs, logger):
    logger.info("{} CONFIGS {}".format('_'*30, "_"*30))
    logger.info("{} Split dataset train - val : {}-{}".format('\t', 1- cfgs.CNN.SPLIT, cfgs.CNN.SPLIT ))
    logger.info("{} ROOT: ".format('\t', cfgs.CNN.ROOT ))
    logger.info("{} DATASET_NAME name: {}".format('\t', cfgs.CNN.DATASET_NAME ))
    logger.info("{} FEATURES: {}".format('\t', cfgs.CNN.FEATURES ))
    logger.info("{} OUTPUT_PATH: {}".format('\t', cfgs.CNN.OUTPUT_PATH ))
    logger.info("{} WEIGHTS_PATH: {}".format('\t', cfgs.CNN.WEIGHTS_PATH ))
    logger.info("{} ARCHITECTURE_MODEL_PATH: {}".format('\t', cfgs.CNN.ARCHITECTURE_MODEL_PATH ))
    logger.info("{} VISUALIZE_LOSS_PATH: {}".format('\t', cfgs.CNN.VISUALIZE_LOSS_PATH ))
    logger.info("{} DISABLE LOG: {}".format('\t', cfgs.CNN.DISABLE_LOG))
    logger.info("{} LOG_PATH: {}".format('\t', cfgs.CNN.LOG_PATH))
    logger.info("{} MODEL: ".format('\t' ))
    logger.info("{} INPUT_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.INPUT_SIZE))
    logger.info("{} OUTPUT_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.OUTPUT_SIZE))
    logger.info("{} ACTIVATION: {}".format('\t\t', cfgs.CNN.MODEL.ACTIVATION))
    logger.info("{} PADDING: {}".format('\t\t', cfgs.CNN.MODEL.PADDING))
    logger.info("{} LEARNING_RATE: {}".format('\t\t', cfgs.CNN.MODEL.LEARNING_RATE))
    logger.info("{} DE_CAY: {}".format('\t\t', cfgs.CNN.MODEL.DE_CAY))
    logger.info("{} LOSS_FUNCTION: {}".format('\t\t', cfgs.CNN.MODEL.LOSS_FUNCTION))
    logger.info("{} METRIC: {}".format('\t\t', cfgs.CNN.MODEL.METRIC))
    logger.info("{} BATCH_SIZE: {}".format('\t\t', cfgs.CNN.MODEL.BATCH_SIZE))
    logger.info("{} EPOCHS: {}".format('\t\t', cfgs.CNN.MODEL.EPOCHS))

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
    print("{} DISABLE_LOG: {}".format('\t', cfgs.CNN.DISABLE_LOG ))
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
    cfgs.CNN.DISABLE_LOG = args.dis_log
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

