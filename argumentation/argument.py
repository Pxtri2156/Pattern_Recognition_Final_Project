import numpy as np
import librosa
import nlpaug

import nlpaug.augmenter.audio as naa

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

### THỊNH 
def crop(data, sampling_rate):
    aug = naa.CropAug(sampling_rate=sampling_rate)
    return aug.augment(data)

def loudness(data):
    aug = naa.LoudnessAug()
    return aug.augment(data)


def mask(data, sampling_rate):
    aug = naa.MaskAug(sampling_rate=sampling_rate, mask_with_noise=False)
    return aug.augment(data)

def speed(data):
    aug = naa.SpeedAug()
    return aug.augment(data)

def vtlp(data, sampling_rate):
    aug = naa.VtlpAug(sampling_rate=sampling_rate)
    return aug.augment(data)

def normalize(data):
    aug = naa.NormalizeAug(method='minmax')
    return aug.augment(data)

def polarity_inversion(data):
    aug = naa.PolarityInverseAug()
    return aug.augment(data)

# taking any example and checking for techniques.
def main():
    # path = np.array(data_path.Path)[1]
    path = "D:/Pattern_Recognition/URDU-Dataset-master/Neutral/SF9_F3_N03.wav"
    data, sample_rate = librosa.load(path,duration=2.5, offset=0.6 )
    print("Shape data before: ", data.shape)
    noise_data = vtlp(data, sample_rate)
    print("Shape noise data: ", noise_data.shape)

if __name__ == "__main__":
    main()