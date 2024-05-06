from keras.applications import VGG19
import torch.nn as nn
import os, sys
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from datasets import StreetSatTrain, StreetSatTest, StreetSatVal

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from main import StreetSatDataModule



class VGG(nn.Module):
    def __init__(self, device, input_shape=(3,128,128)):
        self.vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape).to(device)

    def call(self, x):
        x = self.vgg.preprocess_input(x)
        features = self.vgg.predict(x).reshape(-1, 512)
        return features
    


if __name__ == '__main__':
    device = torch.device('cuda')
    # initialize model
    model = VGG(device=device)
    print("Initialized Model")

    train_cfg = {'target': 'data.datasets.StreetSatTrain'}
    test_cfg = {'target': 'data.datasets.StreetSatTest'}
    val_cfg = {'target': 'data.datasets.StreetSatVal'}

    to_dir = 'data/val/satemb_vgg'
    data_loader = StreetSatDataModule(8, val=val_cfg, train=train_cfg, num_workers=1)
    data_loader.setup()
    samples = data_loader._val_dataloader()
    print("Initialized DataLoader")

    # load data from dataloader
    for batch in tqdm(samples):
        lats, lngs, sat_imgs = batch['latitude'], batch['longitude'], batch['satellite_image']
        inds = []
        outpaths = []
        for i, (lat, lng) in enumerate(zip(lats, lngs)):
            filename = f"{lat},{lng}_satemb_vgg.pt"
            outpath = os.path.join(to_dir, filename)

            if os.path.exists(outpath):
                continue

            inds.append(i)
            outpaths.append(outpath)

        inds = torch.tensor(inds).int()
        sat_imgs = torch.index_select(sat_imgs, 0, inds)

        if len(sat_imgs) == 0:
            continue

        imgs = imgs.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).to(device)

        feature_maps = model(imgs)
        feature_maps = feature_maps.to(torch.float16)

        for outpath, feature_map in zip(outpaths, feature_maps):
            torch.save(feature_map.clone(), outpath)