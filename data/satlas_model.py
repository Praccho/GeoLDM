# pip install satlaspretrain-models
import satlaspretrain_models

import os, sys
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from datasets import StreetSatTrain, StreetSatTest, StreetSatVal

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from main import StreetSatDataModule

class Satlas:
    def __init__(self):
        # initialize a Weights instance
        self.weights_manager = satlaspretrain_models.Weights()

        # initialize Swin-v2-Base model for single images, RGB
        # fpn = feature pyramid network to combine coarse and fine grained representations
        self.model = self.weights_manager.get_pretrained_model(model_identifier="Aerial_SwinB_SI", fpn=True, device='cpu')

    def feature_map(self, sat_img):
        # retrieving second feature map so outputted size is 16x16
        return self.model(sat_img)[1]
    

if __name__ == '__main__':
    # initialize model
    model = Satlas()
    print("Initialized Model")

    train_cfg = {'target': 'data.datasets.StreetSatTrain'}
    test_cfg = {'target': 'data.datasets.StreetSatTest'}
    val_cfg = {'target': 'data.datasets.StreetSatVal'}

    # initialize destination directories
    to_dir = f'val/sat_embeds'
    data_loader = StreetSatDataModule(128, val=val_cfg)
    data_loader.setup()
    samples = data_loader._val_dataloader()
    print("Initialized DataLoader")

    # load data from dataloader
    for batch in tqdm(samples):
        lats, lngs, sat_imgs = batch['latitude'], batch['longitude'], batch['satellite_image']
        mask = torch.ones(len(lats)).long()
        outpaths = []
        for i, (lat, lng) in enumerate(zip(lats, lngs)):
            filename = f"{lat},{lng}_satemb.png"
            outpath = os.path.join(to_dir, filename)
            if os.path.exists(outpath):
                mask[i] = 0
                continue
            outpaths.append(outpath)
            
        
        sat_imgs = sat_imgs[mask]

        if len(sat_imgs) == 0:
            continue

        # expected input is an Aerial (0.5-2m/px high-res imagery)
        # The 0-255 pixel values should be divided by 255 so they are 0-1.
        norm_imgs = ((sat_imgs + 1) * 127.5) / 255.0
        feature_maps = model.feature_map(norm_imgs)

        for outpath, feature_map in zip(outpaths, feature_maps):
            feature_map.save(outpath)