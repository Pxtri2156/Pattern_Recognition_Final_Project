import librosa  
import argparse

def load_data(root_data, name_data):
    '''
    return data_path 
    '''
    pass

def method1():
    pass

def method2():
    pass

def extract_all_features(data):
    '''
    Return: Features vector 
    '''
    # Get data
    # Extract each features(MFCC, F0, .......)
        ## Method 1
        ## Method 2
        ## .......
        ## Method n
    # Concante vector with np.hstack()
    # Return vector 
    pass

def get_features(root_data, data_n):
    # Generate data path 
    # Load data with librosa
    # Argument data
    # Extract default data and argument data 
    # Concante vector with np.vstack()
    # Return result
    pass

def save_features(features, output):
    '''
    Parameter: 
        + features: vector of features 
        + output: the save path 
    Result: save Features vector into file
    '''
    pass

def main(args):
    print(args.root)
    print(args.type)
    print(args.argu)
    print(args.output)

def args_parser():

    parser = argparse.ArgumentParser(description='Argument of extract data')
    parser.add_argument('--root', '-r', required=True, 
                        type=str, help='The path consists all datasets')
    parser.add_argument('--data_n', '-n', default='all', 
                        type=str, help='name of dataset you want to extract')
    parser.add_argument('--type', '-t', default="all", 
                    type=str, help='The type method that you want to extract')
    parser.add_argument('--argu', '-a', action='store_true', 
                    help='The type method that you want extract')                           
    parser.add_argument('--output', '-o', default = "Null", 
                        type=str, help='The root path will save path list and emotion list')
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    main(args)
    