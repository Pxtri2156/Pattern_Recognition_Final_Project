import librosa  
import argparse
import numpy as np 
import sys 
import soundfile as sf

from tqdm import tqdm
sys.path.append("./")
from features_extraction.extract_methods import m_mfcc, m_zcr, m_spectral_rolloff, \
                                                m_chroma_stft, m_rms, m_mel_spectogram, \
                                                m_chroma_cens, m_spectral_centroid, m_spectral_bandwidth, \
                                                m_tempogram, m_spectral_contrast, m_spectral_flatness, custom_mfcc

from argumentation.argument import noise, stretch, shift, pitch

from data_processing.generate_data_path import *

def load_data(root_data, name_data):
    '''
    return data_path 
    '''
    pass



def extract_all_features(data, sample_rate):
    '''
    Return: Features vector 
    '''
    # Get data
    result = np.array([])

    # Extract each features(MFCC, F0, .......)
    ## MFCC
    mfcc = m_mfcc(data, sample_rate)
    # print("shape mfcc: ", mfcc.shape)
    ## Chroma STFT
    sftf = m_chroma_stft(data, sample_rate)
    # print("shape stft: ", sftf.shape)
    ## Zcr
    zcr = m_zcr(data)
    # print('shape zcr: ', zcr.shape)
    ## Spectral Rolloff
    rolloff = m_spectral_rolloff(data, sample_rate)
    # print('shape rolloff: ', rolloff.shape)
    ## Mel Spectogram
    mel = m_mel_spectogram(data, sample_rate)
    # print("shape mel: ", mel.shape)
    ## rms 
    rms = m_rms(data)
    # print("shape rms: ", rms.shape)
    # cens 
    cens = m_chroma_cens(data, sample_rate)
    # print("cens shape: ", cens.shape) 
    # spectral centroid
    centroid = m_chroma_cens(data, sample_rate)
    # print('centroid shape: ', centroid.shape)
    # 
    contrast = m_spectral_contrast(data, sample_rate)
    # print('contrast shape: ', contrast.shape)
    # 
    flatness = m_spectral_flatness(data)
    # print('flatness shape: ', flatness.shape)
    # 
    tempogram = m_tempogram(data, sample_rate)
    # print("tempogram shape: " ,tempogram.shape)
    # Hstack 
    result = np.hstack((result, mfcc, sftf, zcr, mel, rms, cens, centroid,
                          tempogram))

    # Return vector 
    return result
def get_feature_single_path_argument_data(path):
    data, sr = sf.read(path)
    features = extract_all_features(data, sr)
    print("shape of features: ", features.shape)
    # Return result
    return features
      

def get_feature_from_single_path_with_mfcc(path, argument=False):
    # Generate data path 
    # Load data with librosa
    data, sample_rate = librosa.load(path, res_type='kaiser_fast', duration = 2.5, sr=22050*2, offset=0.5)
    print("shape data: ", data.shape)
    features = custom_mfcc(data, sample_rate)
    # Argument data
    if argument == True:
        print("argumented: Noise, Pitch, Shift")
        noise_data = noise(data)
        # print("noise shape: ", noise_data.shape)
        noise_features = custom_mfcc(noise_data, sample_rate)
        # stretch_data = stretch(data)
        pitch_data = pitch(data, sample_rate)
        # print("pitch shape: ", pitch_data.shape)
        pitch_features = custom_mfcc(pitch_data, sample_rate)

        shift_data = shift(data)
        # print("shift_data", shift_data.shape)
        shift_feature = custom_mfcc(shift_data, sample_rate)

        # Concante vector with np.vstack()
        features = np.vstack((features, noise_features, pitch_features , shift_feature))
    # print('features: ', features)
    print("shape of features: ", features.shape)
    # Return result
    return features

def get_feature_from_single_path(path, argument=False):
    # Generate data path 
    # Load data with librosa
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    print("shape data: ", data.shape)
    features = extract_all_features(data, sample_rate)
    # Argument data
    if argument == True:
        print("argumented: Noise, Pitch, Shift")
        noise_data = noise(data)
        # print("noise shape: ", noise_data.shape)
        noise_features = extract_all_features(noise_data, sample_rate)
        # stretch_data = stretch(data)
        pitch_data = pitch(data, sample_rate)
        # print("pitch shape: ", pitch_data.shape)
        pitch_features = extract_all_features(pitch_data, sample_rate)

        shift_data = shift(data)
        # print("shift_data", shift_data.shape)
        shift_feature = extract_all_features(shift_data, sample_rate)

        # Concante vector with np.vstack()
        features = np.vstack((features, noise_features, pitch_features , shift_feature))
    # print('features: ', features)
    print("shape of features: ", features.shape)
    # Return result
    return features

def get_features_from_multi_paths(paths, type_method, data_name, argument=False):
    all_features = np.array([])
    # dem = 0
    errors_index = []
    for i in tqdm(range(len(paths))):
        print("Processing: ", paths[i])
        # if dem > 5:
        #     break
        # dem += 1
        
        try:
          if data_name == "argument_data":
            feature = get_feature_single_path_argument_data(paths[i])
          else:
            if type_method == "all":
              feature = get_feature_from_single_path(paths[i], argument)
            elif type_method == "MFCC":
              print("Use MFCC")
              feature = get_feature_from_single_path_with_mfcc(paths[i], argument)
        except:
          print("Except ")
          errors_index.append(i)
          continue
        if i != 0:
            all_features = np.vstack((all_features, feature))
        else:
            all_features = feature
    return all_features, errors_index

def filter_wrong_features(errors_index, label_lst, argument=False):
    label_lst = np.array(label_lst)
    index_remove = []
    if argument == True:
        for i in errors_index:
            index_remove  += [i*4 + k for k in range(4)]
    else:
        index_remove = errors_index
    print("index remove: ", index_remove)
    label_lst = np.delete(label_lst, index_remove)
    
    return label_lst 



def get_all_features(root_data, data_n, type_method, argument=False):
    # Generate data path 
    path_lst, label_lst = generate_data_path(root_data, data_n)
    # Load data with librosa 
    # Get labels
    if argument == True:
        label_lst = np.repeat(label_lst,4)

    # Get features from paths list
    all_features, errors_index = get_features_from_multi_paths(path_lst, type_method,data_n, argument)
    label_lst = filter_wrong_features(errors_index,label_lst, argument )
    print('all features shape: ', all_features.shape)
    print("all label_lst shape: ", len(label_lst))
    return all_features, label_lst

def save_features(features, label_lst, output,data_n):
    '''
    Parameter: 
        + features: vector of features 
        + output: the save path 
    Result: save Features vector into file
    '''
    save_path = os.path.join(output,data_n)
    if not  os.path.isdir(save_path):
        os.mkdir(save_path)
        print("Created ", save_path)

    feature_path = os.path.join(save_path, 'features.npz')
    label_path = os.path.join(save_path, 'label.txt')
    label_file = open(label_path, 'w')

    np.savez_compressed(feature_path, features )
    print("Saved all of the features")
    for label in label_lst:
         label_file.write("%s\n" % label)   
    print("Saved all of the labels")

def main(args):
    print(args.root)
    print(args.type_method)
    print(args.argu)
    print(args.output)
    print(args.data_n)
    all_featues, label_lst = get_all_features(args.root, args.data_n,args.type_method, args.argu)
    save_features(all_featues, label_lst, args.output, args.data_n)
    # Load data 


def args_parser():

    parser = argparse.ArgumentParser(description='Argument of extract data')
    parser.add_argument('--root', '-r', required=True, 
                        type=str, help='The path consists all datasets')
    parser.add_argument('--data_n', '-n', default='all', 
                        type=str, help='name of dataset you want to extract')
    parser.add_argument('--type_method', '-t', default="all", 
                    type=str, help='The type method that you want to extract')
    parser.add_argument('--argu', '-a', action='store_true', 
                    help='The type method that you want extract')                           
    parser.add_argument('--output', '-o', default = "Null", 
                        type=str, help='The root path will save path list and emotion list')
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    main(args)

'''
 -r E:/Courses/Recognition/Final_Project/Dataset  -n TESS  -o E:/Courses/Recognition/Final_Project/Pattern_Recognition_Final_Project/feature_data  -a 
   
 '''
    