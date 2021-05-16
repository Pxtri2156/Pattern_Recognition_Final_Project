import os
import argparse

from tqdm import tqdm


def generate_DS1(data_path):
    print("\tProcessing DS1")
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

def generate_DS2(data_path):
    print("\tProcessing DS2")
    path_lst = []
    emotion_lst = []
    for sub in tqdm(os.listdir(data_path)):
        sub_path = os.path.join(data_path, sub)
        emotion = sub[-1]
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            path_lst.append(file_path)
            emotion_lst.append(emotion)
    print("Done DS2")
    return path_lst, emotion_lst


def generate_data_path(root_path, nameDB):

    # Generate one datasets
    path_lst = []
    emotion_lst = []
    path_data = os.path.join(root_path, nameDB)
    if nameDB == "DS1":
        path_lst, emotion_lst = generate_DS1(path_data)
    elif nameDB == "DS2":
        path_lst, emotion_lst = generate_DS2(path_data)
    else:
        print("Wrong name dataset !!!")

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

def write_result(output_path, root_path, all_path_lst, all_emotion_lst):
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
    
    

def main(args):
    print(args.root)
    print(args.output)
    all_path_lst, all_emotion_lst =  generate_all_ds(args.root)
    print("all path list: ", all_path_lst)
    print("all emotion list: ", all_emotion_lst)
    if args.output != "Null":
        write_result(args.output, args.root, all_path_lst, all_emotion_lst)

def arg_parser():
    parser = argparse.ArgumentParser(description='Argument of generate data path')
    parser.add_argument('--root', '-r', required=True, 
                        type=str, help='The path consists all datasets')
    parser.add_argument('--output', '-o', default = "Null", 
                        type=str, help='The path save path list and emotion list')

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)

