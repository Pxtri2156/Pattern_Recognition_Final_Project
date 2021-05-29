import librosa  
import argparse
import numpy as np 
import sys 
sys.path.append("./")
from features_extraction.extract_methods import m_mfcc, m_zcr, m_spectral_rolloff, \
                                                m_chroma_stft, m_rms, m_mel_spectogram, \
                                                m_chroma_cens, m_spectral_centroid, m_spectral_bandwidth, \
                                                m_tempogram, m_spectral_contrast, m_spectral_flatness 

from argumentation.argument import noise, stretch, shift, pitch

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
    # spectral centroid
    centroid = m_chroma_cens(data, sample_rate)
    # 
    contrast = m_spectral_contrast(data, sample_rate)
    # 
    flatness = m_spectral_flatness(data)
    # 
    tempogram = m_tempogram(data, sample_rate)
    # Hstack 
    result = np.hstack((result, mfcc, sftf, zcr, rolloff, mel, rms, cens, centroid,
                         contrast, flatness, tempogram))

    # Return vector 
    return result

def get_feature(path, argument=False):
    # Generate data path 
    # Load data with librosa
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_all_features(data, sample_rate)
    # Argument data
    if argument == True:
        print("argumented")
        noise_data = noise(data)

        stretch_data = stretch(data)

        pitch_data = pitch(data)

        shift_data = shift(data)

    # Concante vector with np.vstack()
    print('features: ', features)
    print("shape of features: ", features.shape)
    # Return result
    return features

def get_all_features(root_data, data_n ):
    # Generate data path 
    # Load data with librosa 
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
    path = 'E:/Courses/Recognition/Final_Project/Dataset\TESS\OAF_angry\OAF_back_angry.wav'
    featues = get_feature(path)
    # Load data 


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

'''
 -r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\dataset 
 '''
    