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
        # dev 
        dev_path = os.path.join(folder_path, 'dev')
        dev_features = []
        dev_labels = []
        if os.path.isdir(dev_path):
            dev_features_path = os.path.join(dev_path,"features.npz")
            dev_labels_path = os.path.join(dev_path, "label.txt")
            dev_dict_data = np.load(dev_features_path)
            dev_features = dev_dict_data['arr_0']
            # Load label 
            dev_label_file = open(dev_labels_path, 'r')
            dev_labels = dev_label_file.readlines()
            dev_labels = [label.strip('\n') for label in dev_labels] 
            dev_label_file.close()   

        # train
        train_path = os.path.join(folder_path, 'train')
        train_features = []
        train_labels = []
        if os.path.isdir(train_path):
            train_features_path = os.path.join(train_path,"features.npz")
            train_labels_path = os.path.join(train_path, "label.txt")
            train_dict_data = np.load(train_features_path)
            train_features = train_dict_data['arr_0']
            # Load label 
            train_label_file = open(train_labels_path, 'r')
            train_labels = train_label_file.readlines()
            train_labels = [label.strip('\n') for label in train_labels] 
            train_label_file.close()   
        else:
            assert'Not folder train'

        return train_features, train_labels, dev_features, dev_labels



def main():
    # Test function is here
    pass

if __name__ == "__main__":
    main()