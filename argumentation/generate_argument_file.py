import librosa
import argparse
import os
import json
import nlpaug
from tqdm import tqdm
import soundfile as sf
from IPython.display import Audio
import librosa.display
# import seaborn as sns
import matplotlib.pyplot as plt
import sys 
sys.path.append('./')
from argumentation.argument import noise, pitch, shift, crop, loudness, mask, \
                                    speed, vtlp, normalize, polarity_inversion


def argument_data(data, sr, argument, output, id):

    argumentation_data = None
    if argument == "noise":
        argumentation_data = noise(data)
    elif argument == "pitch":
        argumentation_data = pitch(data, sr)
    elif argument == "shift":
        argumentation_data = shift(data)
    elif argument == "crop":
        argumentation_data = crop(data, sr)
    elif argument == "loudness":
        argumentation_data = loudness(data)
    elif argument == "mask":
        argumentation_data = mask(data,sr)
    elif argument == "speed":
        argumentation_data = speed(data)
    elif argument == "vtlp":
        argumentation_data = vtlp(data,sr)
    elif argument == "normalize":
        argumentation_data = normalize(data)
    elif argument == "polarity_inversion":
        argumentation_data = polarity_inversion(data)
  
    name_file = argument + "_" + str(id) + ".wav"
    save_path = os.path.join(output, name_file)
    # print('save path: ', save_path)
    sf.write(save_path, argumentation_data, sr, "PCM_24")
    print("Saved at {}".format(save_path))
def read_json(path):
  fi = open(path)
  path_dic = json.load(fi)
  fi.close()
  return path_dic 

def generate_one_label(path_list, argument_list, label, output_root):
    output = os.path.join(output_root, label)
    if not os.path.isdir(output):
        print("Create folder: {}".format(output))
        os.mkdir(output)
    id = 0
    for path in tqdm(path_list):
        print("[INFO] Label: {}, id: {} \n\tpath: {}".format(label, id, path))
        data, sr =  librosa.load(path,duration=2.5, offset=0.6 )
        for argument in argument_list:
            argument_data(data, sr, argument, output, id)
        id += 1

def generate_all_label(input_path ,argument_list, output_root):
    path_dic = read_json(input_path)
    for label in path_dic.keys():
        generate_one_label(path_dic[label], argument_list, label, output_root)
    print("Done")

def arg_parser():
    parser = argparse.ArgumentParser(description='Argument of extract data')
    parser.add_argument('--input_path', '-i', required=True, 
                        type=str, help='The path save path json',) 
    parser.add_argument('--output_root', '-o', required=True, 
                        type=str, help='The path will save arugment file')
    parser.add_argument('--list_argument', '-la', required=True, 
                        type=str, help='list argumentation',nargs='+')
    return parser.parse_args()

def main(args):
    print("input path: ", args.input_path)
    print("output path: ", args.output_root)
    print("argument: ", args.list_argument)
    generate_all_label(args.input_path, args.list_argument, args.output_root)

if __name__ == "__main__":
    args = arg_parser()
    main(args)
