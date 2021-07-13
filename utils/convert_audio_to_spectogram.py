import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
#for loading and visualizing audio files
import librosa
import librosa.display
from tqdm import tqdm
import soundfile as sf

def convert_audio_to_spectogram_img(input_path, output_path):
    # data, sr = librosa.load(input_path, sr=44100)
    data, sr = sf.read(input_path)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.savefig(output_path)
    del(data)
    del(X)
    del(Xdb)
    print("Saved image at {}".format(output_path))

def convert_all_audio(input_root, output_root, start, end):
    # df = pd.DataFrame(data=os.listdir(input_root),columns=['name_file'])
    # df.to_csv("link.csv", index=False)
    link_file = pd.read_csv("/content/Pattern_Recognition_Final_Project/datasets/IEMOCAP/link.csv")
    
    if end > len(link_file):
      end = len(link_file) + 1
    for file_name in tqdm(link_file["name_file"][start:end]):
      # file_name = "Ses02F_impro06_M005_neu.wav"
      print("Processing {}".format(file_name))
      input_path = os.path.join(input_root,file_name)
      output_path = os.path.join(output_root,file_name.split(".")[0] + ".png" )
      convert_audio_to_spectogram_img(input_path,output_path )
    print("Done:")
def main(args):
    convert_all_audio(args.root_input, args.root_output, args.start,args.end) 

def args_parser():
    parser = argparse.ArgumentParser(description='Convert audio to spectogram image')
    parser.add_argument('--root_input', '-i', required=True, 
                        type=str, help='The path save all audio file')
    parser.add_argument('--root_output', '-o', default='./ouput', 
                        type=str, help='The path save spectogram image')
    parser.add_argument('--start', '-s', required=True, type=int)
    parser.add_argument('--end', '-e', required=True, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    main(args)
