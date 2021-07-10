import numpy as np
import librosa


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

def save_audio(output, argument_data):
    ## Save argument data
    pass


# taking any example and checking for techniques.
def main():
    # path = np.array(data_path.Path)[1]
    path = "E:/Courses/Recognition/Final_Project/Dataset/CREM/1001_DFA_ANG_XX.wav"
    data, sample_rate = librosa.load(path,duration=2.5, offset=0.6 )
    print("Shape data before: ", data.shape)
    noise_data = noise(data)
    print("Shape noise data: ", noise_data.shape)

if __name__ == "__main__":
    main()