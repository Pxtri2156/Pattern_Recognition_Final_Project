'''
Input: 
    + Name method model
    + Name dataset
    + Argument: True or False
    + Other 

Output: 
    + Model

'''
import argparse

def load_features(features):
    '''
    return data
    Process:
        + Choose dataset
        + Load features from file 
        + Return data
    '''
    pass

def split_data(ratio, data):
    '''
    return train data and test data
    '''
    pass 

def train():

    # Choose dataset
    # Load feature data
    # Split data
    # Choose model
    # Training 
    # Save model 
    pass

def main(args):
    pass

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


if __name__ == "__main__":
    args = arg_parser()
    main(args)