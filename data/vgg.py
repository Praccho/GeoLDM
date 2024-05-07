from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
import os, sys
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from data.datasets import StreetSatTrain, StreetSatTest, StreetSatVal
import tensorflow as tf
import torch.nn as nn
import numpy as np

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from main import StreetSatDataModule



class VGG(nn.Module):
    def __init__(self, input_shape=(128,128,3)):
        super().__init__()
        self.vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    def forward(self, x):
      with tf.device('/device:gpu:1'):
        x = preprocess_input(x)
        features = self.vgg(x)
      
      return features


if __name__ == '__main__':
    # initialize model
    model = VGG()
    print("Initialized Model")

    train_cfg = {'target': 'data.datasets.StreetSatTrain'}
    test_cfg = {'target': 'data.datasets.StreetSatTest'}
    val_cfg = {'target': 'data.datasets.StreetSatVal'}

    to_dir = '/data/train/satemb_vgg'
    data_loader = StreetSatDataModule(256, val=val_cfg, train=train_cfg, num_workers=1)
    data_loader.setup()
    samples = data_loader._train_dataloader()
    print("Initialized DataLoader")

    # load data from dataloader
    for batch in tqdm(samples):
        lats, lngs, sat_imgs = batch['lat'], batch['lng'], batch['satellite_image']
        inds = []
        outpaths = []
        for i, (lat, lng) in enumerate(zip(lats, lngs)):
            filename = f"{lat},{lng}_satemb_vgg.pt"
            outpath = os.path.join(to_dir, filename)

            if os.path.exists(outpath):
                continue

            inds.append(i)
            outpaths.append(outpath)

        inds = np.array(inds, dtype=int)
        sat_imgs = tf.cast(np.take(sat_imgs, inds, axis=0), dtype=tf.float16)

        if len(sat_imgs) == 0:
            continue


        feature_map = model(sat_imgs).numpy()
        feature_map = torch.tensor(feature_map, dtype=torch.float16)

        for outpath, feature_map in zip(outpaths, feature_map):
            torch.save(feature_map.clone(), outpath)