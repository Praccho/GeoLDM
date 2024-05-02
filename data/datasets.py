import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class StreetSatBase(Dataset):
    def __init__(self, root, size=None, transform=None):
        """
        Params:
        
        root: str
            should be "data/[train/test/val]" 
        transforms: func
            any preproc applied to street image
            
        """
        self.root = root
        self.street_dir = os.path.join(root, 'street')
        self.satellite_dir = os.path.join(root, 'satellite')
        self.transform = transform
        self.size = size
        self.resize = transforms.Resize((size, size)) if size else None

        self.image_files = [f for f in os.listdir(self.street_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        street_img_name = os.path.join(self.street_dir, self.image_files[idx])
        lat_lng = self.image_files[idx].split('_')[0]
        satellite_img_name = os.path.join(self.satellite_dir, f'{lat_lng}_sat.png')

        street_img = Image.open(street_img_name)
        satellite_img = Image.open(satellite_img_name)

        # cropping satellite image
        SAT_IMG_OUT_SIZE = 256
        left = (street_img.width - SAT_IMG_OUT_SIZE) / 2
        top = (street_img.height - SAT_IMG_OUT_SIZE) / 2
        right = (street_img.width + SAT_IMG_OUT_SIZE) / 2
        bottom = (street_img.height + SAT_IMG_OUT_SIZE) / 2

        if self.transform:
            satellite_img = self.transform(satellite_img)

        satellite_img = satellite_img.crop((left, top, right, bottom))

        if self.resize:
            street_img = self.resize(street_img)

        street_img = np.array(street_img).astype(np.uint8)
        street_img = (street_img / 127.5 - 1.0).astype(np.float32)

        satellite_img = np.array(satellite_img).astype(np.uint8)
        satellite_img = (satellite_img / 127.5 - 1.0).astype(np.float32)

        sample = {'latitude': float(lat_lng.split(',')[0]), 'longitude': float(lat_lng.split(',')[1]), 
                  'street_image': street_img, 'satellite_image': satellite_img}

        return sample
    
class StreetSatTrain(StreetSatBase):
    def __init__(self, **kwargs):

        transform_satellite = transforms.Compose([
            transforms.RandomRotation(360),
        ])

        super().__init__(root='data/train', transform=transform_satellite, **kwargs)

class StreetSatVal(StreetSatBase):
    def __init__(self, **kwargs):
        super().__init__(root='data/val', **kwargs)

class StreetSatTest(StreetSatBase):
    def __init__(self, **kwargs):
        super().__init__(root='data/test', **kwargs)