import os 
import librosa                                                                                                                                                                                                                                                                                                                                                             
import argparse
import numpy as np 
import sys
import joblib
import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import warnings
sys.path.append("./")
from utils.loading import load_features_labels
from utils.config_parse import get_config
from features_extraction.extract_featrures import get_features_from_multi_paths

def evaluate_model(y_true, y_pred, tar_names):

    cls_report = classification_report(y_true, y_pred, target_names=tar_names )
    print("{} RESULT EVALUATION {} \n {}".format('_'*30, "_"*30, cls_report ))


def predict(args, cfgs):
    Cls_name = ['A', 'B', 'C']
    print("[INFO]: Preparing data")
    paths = glob.glob(args.in_audios +  "/*.wav")
    print("[INFO]: Exacting data")
    features = get_features_from_multi_paths(paths)
    print("[INFO]: Loading model")
    model_path = os.path.join(args.model)
    model = joblib.load(model_path)
    print("[INFO]: Predictings")
    predited = model.predict(features)
    print("[INFO]: Saving result")
    print('[INFO]: Done')

def main(args, cfgs):
    predict(args, cfgs)
def arg_parser():
    parser = argparse.ArgumentParser(description='Argument of extract data')
    parser.add_argument('--model', '-m', required=True, 
                        type=str, help='The path saved model')
    parser.add_argument('--in_audios', '-i', required=True, 
                        type=str, help='The path contain file audio you want to predict')
    parser.add_argument('--output', '-o', default= "./predited_result", 
                        type=str, help='The path will save predicted')
    parser.add_argument('--configs_file', '-c', required=True, 
                        type=str, help='The path config file for model')
    return parser.parse_args()

def print_args_cfg(args, cfgs):
    print("{} ARGUMENT {}".format('_'*30, "_"*30))
    print("{} Model: \n\t{}".format('#'*3, args.model))
    print("{} The path of input file audio : \n\t{}".format('#'*3, args.in_audios))
    print("{} The path of output predict : \n\t{}".format('#'*3, args.output))
    print("{} The path of configs file : \n\t{}".format('#'*3, args.configs_file))

    print("{} CONFIGS {}".format('_'*30, "_"*30))
    print("{} Split dataset train - val : {}-{}".format('#'*3, 1- cfgs.KNN.SPLIT, cfgs.KNN.SPLIT ))
    print("{} Number neighbor: {}".format('#'*3, cfgs.KNN.K))
    print("{}\n".format('_'*100))


if __name__ == "__main__":
    args = arg_parser()
    cfgs = get_config()
    cfgs.merge_from_file('./configs/BASE/Base.yaml')
    if not os.path.isfile(args.configs_file):
        print("File configs not exits!!")
    cfgs.merge_from_file(args.configs_file)
    print_args_cfg(args, cfgs)
    main(args, cfgs)

'''
    -m E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\models\KNN\model.pkl \
    -i E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\dataset\test \
    -o tri \
    -c E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\KNN\KNN.yaml

    -m E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\models\KNN\model.pkl -i E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\dataset\test -o tri -c E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\KNN\KNN.yaml
'''