import os
import argparse
import numpy as np
from tqdm import tqdm
import sys 
import glob
sys.path.append("./")
# from config import TYPE_DS1, TYPE_DS2

def generate_ravdess(data_path):
    print("\tProcessing RAVDESS")
    path_lst = []
    emotion_lst = []
    for file in tqdm(os.listdir(data_path)):
        file_path = os.path.join(data_path, file)
        print("file path: ", file_path)
        path_lst.append(file_path)

        emotion = file.split(".")[0][-1]
        emotion_lst.append(emotion)
    print("Done DS1")
    return path_lst, emotion_lst

def generate_tess(data_path):
    print("\tProcessing dataset TESS")
    path_lst = []
    emotion_lst = []
    for sub in tqdm(os.listdir(data_path)):
        print("Processing sub folder: ", sub)
        sub_path = os.path.join(data_path, sub)
        emotion = sub.split("_")[-1].lower()
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            path_lst.append(file_path)
            emotion_lst.append(emotion)
    print("Done DS2")
    return path_lst, emotion_lst

def generate_savee(data_path):
    pass

def generate_crem(data_path):
    print("\tProcessing dataset CREM")
    path_lst = []
    emotion_lst = []
    for file in tqdm(os.listdir(data_path)):
        file_path = os.path.join(data_path, file)
        print("file path: ", file_path)
        path_lst.append(file_path)

        emotion = file.split(".")[0].split("_")[-2]
        emotion_lst.append(emotion)
    print("Done")
    return path_lst, emotion_lst

def generate_meld(data_path):
    print("\tProcessing dataset MELD")
    path_lst = []
    emotion_lst = []
    for file in tqdm(os.listdir(data_path)):
        file_path = os.path.join(data_path, file)
        print("file path: ", file_path)
        path_lst.append(file_path)

        emotion = file.split(".")[0].split("_")[-1]
        emotion_lst.append(emotion)
    print("Done")
    return path_lst, emotion_lst

def generate_iemocap(npy_path):
    print("Preprocessing IEMOCAP")
    path_lst = []
    emotion_lst = []
    npy_lst = glob.glob(npy_path + "/*.npy")
    for folder_path in npy_lst:
        data = np.load(folder_path)
        for fi_path in data:
            path_lst.append(fi_path)
            emotion_lst.append(fi_path.split('.')[0].split("_")[-1])
    return path_lst, emotion_lst

def generate_urdu(npy_path):
    print("Processing URDU")
    path_lst = []
    emotion_lst = []
    npy_lst = glob.glob(npy_path + "/*.npy")
    for folder_path in npy_lst:
        data = np.load(folder_path)
        for fi_path in data:
            path_lst.append(fi_path)
            emotion_lst.append(fi_path.split('/')[3])
    return path_lst, emotion_lst

def generate_data_path(root_path, nameDB):

    # Generate one datasets

    path_lst = []
    emotion_lst = []
    path_data = os.path.join(root_path, nameDB)
    if nameDB == "RAVDESS":
        path_lst, emotion_lst = generate_ravdess(path_data)
    elif nameDB == "TESS" :
        path_lst, emotion_lst = generate_tess(path_data)
    elif nameDB == "MELD":
        path_lst, emotion_lst = generate_meld(path_data)
    elif nameDB == "IEMOCAP":
        path_lst, emotion_lst = generate_iemocap(path_data)
    elif nameDB == "URDU":
        path_lst, emotion_lst = generate_urdu(path_data)
    elif nameDB == "CREM":
        path_lst, emotion_lst = generate_crem(path_data)
    else:
       assert(False),"Wrong name dataset !!!"

    return path_lst, emotion_lst

def generate_all_ds(root_path):
    # Generate many datasets
    all_path_lst = []
    all_emotion_lst = []
    name_lst = os.listdir(root_path)
    for name in tqdm(name_lst):
        path_lst, emotion_lst = generate_data_path(root_path, name)
        all_path_lst = all_path_lst + path_lst
        all_emotion_lst = all_emotion_lst + emotion_lst
    print("Done all datasets")
    return all_path_lst, all_emotion_lst

def write_all_result(output_path, root_path, all_path_lst, all_emotion_lst):
    name_lst = os.listdir(root_path)
    folder_n = 'DB'
    for name in name_lst:
        folder_n = folder_n + '_' + name
    folder_path = os.path.join(output_path, folder_n)

    if not  os.path.isdir(folder_path):
        os.mkdir(folder_path)
        print("Create: ", folder_path)
    
    label_p = os.path.join(folder_path, "labels.txt")
    path_p = os.path.join(folder_path, 'paths.txt')

    label_f = open(label_p, "w")
    path_f = open(path_p, "w")
    for i in range(len(all_emotion_lst)):
        # print("bug: ", all_path_lst[i])
        path_f.write(all_path_lst[i] + '\n')
        label_f.write(all_emotion_lst[i] + "\n")

    label_f.close()
    path_f.close()

def write_result(output_path, name_DS, all_path_lst, all_emotion_lst):
    folder_path = os.path.join(output_path, name_DS)
    if not  os.path.isdir(folder_path):
        os.mkdir(folder_path)
    print("Created: ", folder_path)

    label_p = os.path.join(folder_path, "labels.txt")
    path_p = os.path.join(folder_path, 'paths.txt')

    label_f = open(label_p, "w")
    path_f = open(path_p, "w")
    for i in range(len(all_emotion_lst)):
        # print("bug: ", all_path_lst[i])
        path_f.write(all_path_lst[i] + '\n')
        label_f.write(all_emotion_lst[i] + "\n")

    label_f.close()
    path_f.close()
    

def main(args):
    print(args.root)
    print(args.output)
    # all_path_lst, all_emotion_lst =  generate_all_ds(args.root)
    all_path_lst, all_emotion_lst  = generate_data_path(args.root, args.nameDB)
    print("all path list: ", all_path_lst[:10])
    print("all emotion list: ", all_emotion_lst[:10])
    if args.output != "Null":
        write_result(args.output, "TESS", all_path_lst, all_emotion_lst)

def arg_parser():
    parser = argparse.ArgumentParser(description='Argument of generate data path')
    parser.add_argument('--root', '-r', required=True, 
                        type=str, help='The path consists all datasets')
    parser.add_argument('--nameDB', '-n', required=True, 
                        type=str, help='Name of datasets ')
    parser.add_argument('--output', '-o', default = "Null", 
                        type=str, help='The path save path list and emotion list')

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)
"-r E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\datasets -n CREM"
