import os 
import numpy as np

def load_features_labels(feature_root, data_n):
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

def main():
    # Test function is here
    pass

if __name__ == "__main__":
    main()