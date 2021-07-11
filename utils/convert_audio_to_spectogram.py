import os
import matplotlib.pyplot as plt
import argparse

#for loading and visualizing audio files
import librosa
import librosa.display

def convert_audio_to_spectogram_img(input_path, output_path):
    data, sr = librosa.load(input_path, sr=44100)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.savefig(output_path)
    print("Saved image at {}".format(output_path))

def convert_all_audio(input_root, output_root):
    for file_name in os.listdir(input_root):
        print("Processing {}".format(file_name))
        input_path = os.path.join(input_root,file_name )
        output_path = os.path.join(output_root,file_name.split(".")[0] + ".png" )
        convert_audio_to_spectogram_img(input_path,output_path )
    print("Done:")
def main(args):
    convert_all_audio(args.root_input, args.root_output) 

def args_parser():
    parser = argparse.ArgumentParser(description='Convert audio to spectogram image')
    parser.add_argument('--root_input', '-i', required=True, 
                        type=str, help='The path save all audio file')
    parser.add_argument('--root_output', '-o', default='./ouput', 
                        type=str, help='The path save spectogram image')
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    main(args)
