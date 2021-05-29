import librosa  
import argparse
import numpy as np 

def m_chroma_stft(data, sample_rate):
    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    return chroma_stft 

def m_chroma_cqt(data, sample_rate):
    cq = np.mean(librosa.feature.chroma_cqt(y=data, sr=sample_rate).T, axis=0)
    # print("cq: ", cq)
    return cq

def m_chroma_cens(data, sample_rate):
    cens = np.mean(librosa.feature.chroma_cens(y=data, sr=sample_rate).T, axis=0)
    # print("cens: ", cens)
    return cens


def m_mel_spectogram(data, sample_rate):
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    # print('mel: ', librosa.feature.melspectrogram(y=data, sr=sample_rate))
    # print("after: ", mel )
    return mel  

def m_mfcc(data, sample_rate):
    # MFCC 
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    # print("mfcc: ",  (librosa.feature.mfcc(y=data, sr=sample_rate)))
    # print("after: ", mfcc )
    return mfcc  

def  m_rms(data):
    # Root Mean Square Value. \
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    # print("rms: ", librosa.feature.rms(y=data))
    # print("after: ", rms)
    return rms

def m_spectral_centroid(data, sample_rate):
    cent = librosa.feature.spectral_centroid(y=data, sr=sample_rate)[-1]
    # print("cent: ", cent)
    return cent

def m_spectral_bandwidth(data, sample_rate):
    bw = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)[-1]
    return bw 

def m_spectral_contrast(data, sample_rate):
    constrast = librosa.feature.spectral_contrast(y=data, sr=sample_rate)[-1]
    return constrast

def m_spectral_flatness(data):
    flatness = librosa.feature.spectral_flatness(y=data)[-1]
    return flatness

def m_spectral_rolloff(data, sample_rate):
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)[-1]
    return rolloff 

def m_poly_features(data, sample_rate):
    pass

def m_tonnetz(data, sample_rate):
    data = librosa.effects.harmonic(data)
    tonnetz = librosa.feature.tonnetz(data = data, sr=sample_rate)

def m_tempogram(data, sample_rate):
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=data, sr=sample_rate, hop_length=hop_length)
    tempogram = np.mean(librosa.feature.tempogram(onset_envelope=oenv, sr=sample_rate,
                                      hop_length=hop_length).T, axis=0)
    return tempogram

def m_fourier_tempogram(data, sample_rate):
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=data, sr=sample_rate, hop_length=hop_length)
    tempogram = np.mean(librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sample_rate,
                                      hop_length=hop_length).T, axis=0)
    return tempogram


def m_zcr(data):
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    # print("zero crossing rate: ", librosa.feature.zero_crossing_rate(y=data))
    # print("after mean: ", zcr)
    return zcr

def pitch(data, sample_rate):
    pitches, magnitudes = librosa.piptrack(y=data, sr=sample_rate)
    return pitches

def main():
    path = 'E:/Courses/Recognition/Final_Project/Dataset\TESS\OAF_angry\OAF_back_angry.wav'
    data, sample_rate= librosa.load(path, duration=2.5, offset=0.6)
    result = pitch(data, sample_rate)
    print(result)

if __name__ == "__main__":
    main()