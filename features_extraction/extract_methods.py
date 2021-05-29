import librosa  
import argparse
import numpy as np 

def m_chroma_stft(data, sample_rate):
    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    print("stft: ", librosa.feature.chroma_stft(S=stft, sr=sample_rate))
    print("after: ", chroma_stft)
    return stft 

def m_chroma_cqt(data, sample_rate):
    pass

def m_chroma_cens():
    pass


def m_mel_spectogram(data, sample_rate):
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    print('mel: ', librosa.feature.melspectrogram(y=data, sr=sample_rate))
    print("after: ", mel )
    return mel  

def m_mfcc(data, sample_rate):
    # MFCC 
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    print("mfcc: ",  librosa.feature.mfcc(y=data, sr=sample_rate))
    print("after: ", mfcc )
    return mfcc  

def  m_rmsv(data):
    # Root Mean Square Value. \
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    print("rms: ", librosa.feature.rms(y=data))
    print("after: ", rms)
    return rms

def m_spectral_centroid():
    pass 

def m_spectral_bandwidth():
    pass

def m_spectral_contrast():
    pass 

def m_spectral_flatness():
    pass 

def m_spectral_rolloff():
    pass 

def m_poly_features():
    pass 

def m_tonnetz():
    pass 

def m_tempogram():
    pass

def m_fourier_tempogram():
    pass 


def m_zcr(data):
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    # print("zero crossing rate: ", librosa.feature.zero_crossing_rate(y=data))
    # print("after mean: ", zcr)
    return zcr

def main():
    path = 'E:/Courses/Recognition/Final_Project/Dataset\TESS\OAF_angry\OAF_back_angry.wav'
    data, sample_rate= librosa.load(path, duration=2.5, offset=0.6)
    m_mel_spectogram(data, sample_rate)

if __name__ == "__main__":
    main()