import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import torch 
from torch.utils.data import Dataset, DataLoader

from utils import *
from classify_image.src.augmentations import *
from classify_image.src.models import Classifier
from classify_image.src.datasets import ICDARDataset

import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


def convert_audio_to_spectogram_img():
    input_path = '/content/audio/Ses01F_impro01_F000_neu.wav'
    output_path = 'test/image_test.png'
    data, sr = sf.read(input_path)
    try:
      X = librosa.stft(data)
      Xdb = librosa.amplitude_to_db(abs(X)) 
      plt.figure(figsize=(14, 5))
      librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
      plt.savefig(output_path)
      del(data)
      del(sr)
      del(X)
      del(Xdb)
    except librosa.util.exceptions.ParameterError:
      print('Error')

def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        image_preds = model(imgs)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def run_infer(root, name_model, image_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # convert_audio_to_spectogram_img()
    device = torch.device('cpu')
    n_classes = 6
    
    tst_preds = []

    TEST_DIR = image_path.split('/')[0]
    test = pd.DataFrame()
    
    test['image_id'] = [image_path.split('/')[-1]]
    
    if "resnet" in name_model: 
        model_arch = 'resnet50'
    elif "vit" in name_model:
        model_arch = 'vit_small_patch32_384'
    elif "nfnet" in name_model:
        model_arch = 'nfnet_f0'
    elif 'effb4' in name_model:
        model_arch = 'tf_efficientnet_b4_ns'
        
    if "vit" in model_arch:
        testset = ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(384))
    elif 'nfne' in model_arch:
        testset = ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(192))
    else:
        testset = ICDARDataset(test, TEST_DIR, transforms=get_inference_transforms(224))
        
    
    tst_loader = DataLoader(
        testset, 
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=False)

    model = Classifier(model_arch, n_classes).to(device)
    model_path = os.path.join(root, name_model, 'best.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    
    with torch.no_grad():
        for _ in range(1):
            tst_preds += [inference_one_epoch(model, tst_loader, device)]
    
    del model

    avg_tst_preds = np.mean(tst_preds, axis=0)

    pred = avg_tst_preds[0]
    s = sum(pred)
    # print(pred)
    result = []
    for i in pred:
      result.append(round(i*100/s,2))
    print(result)
    metrics = {
            "classifier":result,
            "spectrum" : image_path
        }
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
     metrics = run_infer(root, name_model, image_path)
