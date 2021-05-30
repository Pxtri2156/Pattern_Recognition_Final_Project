'''
Input: 
    + Name method model
    + Name dataset
    + Argument: True or False
    + Other 

Output: 
    + Model

'''
import os 
import librosa                                                                                                                                                                                                                                                                                                                                                             
import argparse
import numpy as np 
import sys 
from tqdm import tqdm
sys.path.append("./")
import argparse

from sklearn.model_selection import train_test_split
from config import SPLIT_RATIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from methods.deep_model import example_deep

def load_features(feature_root, data_n):
    '''
    return data
    Process:
        + Choose dataset 
        + Load features from file 
        + Return data   
    '''
    folder_path = os.path.join(feature_root, data_n)
    if not os.path.isdir(folder_path):
        print("Wrong dataset name !!!!")
    else:
        feature_path = os.path.join(folder_path, 'features.npz')
        label_path = os.path.join(folder_path, 'label.txt')
        # Load features
        dict_data = np.load(feature_path)
        features = dict_data['arr_0']
        # Load label 
        label_f = open(label_path, 'r')
        labels = label_f.readlines()
        labels = [label.strip('\n') for label in labels] 
        label_f.close()   

        return features, labels        

        
def choose_model(name, cls_n,  *args):
    model = None
    if name == "KNN":
        K = len(cls_n)
        model = KNeighborsClassifier(n_neighbors=K)
    elif name == "SVM":
        model = svm.SVC(kernel="linear")
    elif name == 'EX_deep':
        model = example_deep()
        model.build
    else:
        print("Wrong model name!!!")
    return model

def train(feature_root, data_n, model_n):

    # Load feature data
    features, labels = load_features(feature_root, data_n)
    # print('features: ', features)
    # print("labels: ", labels)
    print('features shape: ', features.shape) 
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    cls_n = le.classes_
    labels = labels[:24]
    print('len of labels: ', len(labels))
    # Split data 

    X_train , X_val, Y_train, Y_val = train_test_split(features, labels, test_size=SPLIT_RATIO, shuffle=True)
    print("X train shape: ", X_train.shape)
    print('X val shape: ', X_val.shape) 
    print('Y train',  Y_train)
    print("Y train shape: ", len(Y_train))
    print("Y val: " , len(Y_val))
    
    # Choose model
    model = choose_model(model_n, cls_n)
    # Training
    model.fit(X_train , X_val, Y_train, Y_val )
    predicted = model.predict(X_val)
    print("predicted: ", le.inverse_transform(predicted))
    
    # Save model kl;;l
def main(args):
    print('root: ', args.root)
    print('dataset name: ', args.data_n)
    print('argument: ', args.argu)
    train(args.root, args.data_n, args.model)

def arg_parser():
    parser = argparse.ArgumentParser(description='Argument of extract data')
    parser.add_argument('--root', '-r', required=True, 
                        type=str, help='The path consists all type extracted features')
    parser.add_argument('--data_n', '-n', default='all', 
                        type=str, help='name of dataset you want to extract')
    parser.add_argument('--features', '-f', default="all", 
                    type=str, help='The type method that you want to extract')
    parser.add_argument('--argu', '-a', action='store_true', 
                    help='The type method that you want extract')
    parser.add_argument('--model', '-m', default="KNN", 
                help='The type method that you want extract')                              
    parser.add_argument('--output', '-o', required=True, 
                        type=str, help='The path will save model')

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)

'''
-r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\feature_data -n TESS -o tri - m KNN
'''