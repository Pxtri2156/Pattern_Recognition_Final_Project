import os 
import librosa                                                                                                                                                                                                                                                                                                                                                             
import argparse
import numpy as np 
import sys 
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
import warnings
sys.path.append("./")
from utils.loading import load_features_labels
from utils.config_parse import get_config


def train(args, cfgs):

    # Load feature data
    print("[INFO]: Step 1/6 - Loading features and labels]")
    features, labels = load_features_labels(args.root, args.data_n)
    # print('features: ', features)
    # print("labels: ", labels)
    print("[INFO]: Step 2/6 - Preprocessing labels]")
    print('features shape: ', features.shape) 
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    cls_n = le.classes_
    labels = labels[:24]
    print('len of labels: ', len(labels))
    # Split data 
    print("[INFO]: Step 3/6 - Split data with train - test : {}-{}".format('#'*3, 1- cfg.KNN.SPLIT, cfg.KNN.SPLIT ))
    X_train , X_val, Y_train, Y_val = train_test_split(features, labels, test_size= cfgs.KNN.SPLIT, shuffle=True)
    print("X train shape: ", X_train.shape)
    print('X val shape: ', X_val.shape) 
    print('Y train',  Y_train)
    print("Y train shape: ", len(Y_train))
    print("Y val: " , len(Y_val))
    # Define model
    print("[INFO]: Step 4/6 Define model KNeighborsClassifier ")
    model = KNeighborsClassifier(n_neighbors = cfgs.KNN.K )
    # Training
    print("[INFO]: Step 5/6 Training ")
    model.fit(X_train , Y_train )
    print("[INFO]: Completed training")
    print("Test on validation set")
    predicted = model.predict(X_val)
    print("predicted: ", le.inverse_transform(predicted))
    # Save model kl;;l
    print("[INFO]: 6/6 Saving model")

def main(args, cfgs):

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
    print("{} Split dataset train - test : {}-{}".format('#'*3, 1- cfg.KNN.SPLIT, cfg.KNN.SPLIT ))
    print("{} Number neighbor: {}".format('#'*3, cfgs.KNN.K))
    print("{}\n".format('_'*100))


if __name__ == "__main__":
    args = arg_parser()
    cfg = get_config()
    cfg.merge_from_file('./configs/BASE/Base.yaml')
    if not os.path.isfile(args.configs_file):
        print("File configs not exits!!")
    cfg.merge_from_file(args.configs_file)
    print_args_cfg(args, cfg)
    main(args, cfg)

'''
-r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\feature_data -n TESS -o tri   -c E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\KNN\KNN.yaml
'''