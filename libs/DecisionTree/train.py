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
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

import warnings
sys.path.append("./")
from utils.loading import load_features_labels
from utils.config_parse import get_config
from utils.write_log import write_config_args_2_log
from utils.setup_log import setup_logging
import logging


def evaluate_model(y_true, y_pred, tar_names):

    cls_report = classification_report(y_true, y_pred, target_names=tar_names )
    print("{} RESULT EVALUATION {} \n {}".format('_'*30, "_"*30, cls_report ))
    logger.info("{} RESULT EVALUATION {} \n {}".format('_'*30, "_"*30, cls_report ))


def train(args, cfgs):

    # Load feature data
    print("[INFO]: Step 1/7 - Loading features and labels]")
    train_features, train_labels, dev_features, dev_labels = load_features_labels(args.root, args.data_n)
    # print('features: ', features)
    # print("labels: ", labels)
    print("[INFO]: Step 2/7 - Preprocessing labels]")
    print('features shape: ', train_features.shape) 
    le = preprocessing.LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    cls_n = le.classes_
    print("class: ", cls_n)
    # train_labels = train_labels[:24] # Must to comment
    print('len of train labels: ', len(train_labels))
    # Split data 
    print("[INFO]: Step 3/7 Split data ")
    if dev_features == []:
        print("[INFO]:  Split data with train - val : {}-{}".format( 1- cfgs.DS.SPLIT, cfgs.DS.SPLIT ))
        X_train , X_val, Y_train, Y_val = train_test_split(train_features, train_labels, test_size= cfgs.DS.SPLIT, shuffle=True)
    else:
        print("[INFO]: Loading dev path")
        dev_labels = le.fit_transform(dev_labels)
        X_train, Y_train = train_features, train_labels
        X_val, Y_val = dev_features, dev_labels

    print("X train shape: ", X_train.shape)
    print('X val shape: ', X_val.shape) 
    print('Y train',  Y_train)
    print("Y train shape: ", len(Y_train))
    print("Y val: " , len(Y_val))
    # Define model
    print("[INFO]: Step 4/7 Define model KNeighborsClassifier ")
    model =  tree.DecisionTreeClassifier()
    # Training
    print("[INFO]: Step 5/7 - Training ")
    model.fit(X_train , Y_train )
    print("[INFO]: Completed training")
    print("[INFO]: Step 6/7 - Evaluation on validation set")
    predicted = model.predict(X_val)
    # Y_val = [1, 2, 7, 3, 0,4,5, 6,6]
    # predicted = [0, 0 ,0 ,0, 0, 0, 0, 1,2]
    # print('Y val: ', Y_val)
    # print("predicted: ", predicted)
    evaluate_model(Y_val, predicted, cls_n)
    # print("predicted: ", le.inverse_transform(predicted))
    # Save model
    print("[INFO]: 6/6 - Saving model")
    logging.info("Model was trained")
    save_path = os.path.join( args.output, 'model.pkl')
    joblib.dump(model, save_path)


def main(args, cfgs):
    global logger 
    logger = setup_logging(os.path.join(args.output, 'log.log'))
    logger.disabled = False;
    write_config_args_2_log(cfgs, args, logger)
    train(args, cfgs)

def arg_parser():
    parser = argparse.ArgumentParser(description='Argument of extract data')
    parser.add_argument('--root', '-r', required=True, 
                        type=str, help='The path consists all type extracted features')
    parser.add_argument('--data_n', '-n', default='all', 
                        type=str, help='name of dataset you want to train')
    parser.add_argument('--features', '-f', default="all", 
                    type=str, help='The type method that you want to extract')
    parser.add_argument('--argu', '-a', action='store_true', 
                    help='Augment data')
    parser.add_argument('--output', '-o', required=True, 
                        type=str, help='The path will save model')
    parser.add_argument('--configs_file', '-c', required=True, 
                        type=str, help='The path config file for model')
    return parser.parse_args()

def print_args_cfg(args, cfgs):
    print("{} ARGUMENT {}".format('_'*30, "_"*30))
    print("{} The path root of exacted features: \n\t{}".format('#'*3, args.root))
    print("{} Dataset name to train: \n\t{}".format('#'*3, args.data_n))
    print("{} The methods exacted feature: \n\t{}".format('#'*3, args.features))
    print("{} Augment data: \n\t{}".format('#'*3, args.argu))
    print("{} Dataset name to train: \n\t{}".format('#'*3, args.data_n))
    print("{} The path of model ouput : \n\t{}".format('#'*3, args.output))
    print("{} The path of configs file : \n\t{}".format('#'*3, args.configs_file))

    print("{} CONFIGS {}".format('_'*30, "_"*30))
    print("{} Split dataset train - val : {}-{}".format('#'*3, 1- cfgs.DS.SPLIT, cfgs.DS.SPLIT ))
    print("{}\n".format('_'*100))


if __name__ == "__main__":
    args = arg_parser()
    cfgs = get_config()
    cfgs.merge_from_file('./configs/BASE/Base.yaml')
    if not os.path.isfile(args.configs_file):
        print("File configs not exits!!")
    # old_stdout = sys.stdout
    # log_file = open(os.path.join(args.output, 'log.log'),"w")
    # sys.stdout = log_file
    cfgs.merge_from_file(args.configs_file)
    print_args_cfg(args, cfgs)
    main(args, cfgs)
    # sys.stdout = old_stdout

    # log_file.close()

'''
    -r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\feature_data \
    -n TESS \
    -o E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\libs\GAUSSIAN  \
    -c E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\GAUSSIAN\GAUSSIAN.yaml

-r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\feature_data -n TESS -o E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\models\GaussianNB  -c E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\GaussianNB\GaussianNB.yaml
'''