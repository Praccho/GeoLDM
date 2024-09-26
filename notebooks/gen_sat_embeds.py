import satlaspretrain_models
import os, sys
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from data.datasets import StreetSatTrain, StreetSatTest, StreetSatVal
from main import StreetSatDataModule


import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

class Satlas:
    def __init__(self):
        # initialize a Weights instance
        self.weights_manager = satlaspretrain_models.Weights()

        # initialize Swin-v2-Base model for single images, RGB
        # fpn = feature pyramid network to combine coarse and fine grained representations
        self.model = self.weights_manager.get_pretrained_model(model_identifier="Aerial_SwinB_SI", fpn=True)

    def feature_map(self, sat_img):
        # retrieving second feature map so outputted size is 16x16
        return self.model(sat_img)[2]
# initialize model
model = Satlas()
print("Initialized Model")

train_cfg = {'target': 'data.datasets.StreetSatTrain'}
test_cfg = {'target': 'data.datasets.StreetSatTest'}
val_cfg = {'target': 'data.datasets.StreetSatVal'}


# initialize destination directories
to_dir = '/content/GeoLDM/data/train/satemb'
data_loader = StreetSatDataModule(256, val=val_cfg, train=train_cfg)
data_loader.setup()
samples = data_loader._train_dataloader()
print("Initialized DataLoader")

# load data from dataloader
for batch in tqdm(samples):
    lats, lngs, sat_imgs = batch['latitude'], batch['longitude'], batch['satellite_image']
    inds = []
    outpaths = []
    for i, (lat, lng) in enumerate(zip(lats, lngs)):
        filename = f"{lat},{lng}_satemb.pt"
        outpath = os.path.join(to_dir, filename)

        if os.path.exists(outpath):
            continue

        inds.append(i)
        outpaths.append(outpath)

    inds = torch.tensor(inds).int()
    sat_imgs = torch.index_select(sat_imgs, 0, inds)

    if len(sat_imgs) == 0:
        continue

    # expected input is an Aerial (0.5-2m/px high-res imagery)
    # The 0-255 pixel values should be divided by 255 so they are 0-1.
    norm_imgs = ((sat_imgs + 1) * 127.5) / 255.0
    norm_imgs = norm_imgs.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)

    feature_maps = model.feature_map(norm_imgs)
    feature_maps = feature_maps.to(torch.float16)

    for outpath, feature_map in zip(outpaths, feature_maps):
        torch.save(feature_map.clone(), outpath)


print(len(os.listdir("/content/GeoLDM/data/train/satemb")))
print(len(os.listdir("/content/GeoLDM/data/train/satellite")))