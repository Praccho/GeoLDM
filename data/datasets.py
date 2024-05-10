import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


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
        self.satellite_emb_dir = os.path.join(root, 'satemb')
        self.satellite_emb_vgg_dir = os.path.join(root, 'satemb_vgg')
        self.transform = transform
        self.size = size
        self.resize = transforms.Resize((size, size)) if size else None

        self.image_files = [f for f in os.listdir(self.street_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        street_img_name = os.path.join(self.street_dir, self.image_files[idx])
        lat_lng = self.image_files[idx].split('_')[0]
        lat, lng = lat_lng.split(',')
        # lat, lng = torch.tensor(lat), torch.tensor(lng)

        satellite_img_name = os.path.join(self.satellite_dir, f'{lat_lng}_sat.png')
        satellite_emb_name = os.path.join(self.satellite_emb_dir, f'{lat_lng}_satemb.pt')
        satellite_emb_vgg_name = os.path.join(self.satellite_emb_vgg_dir, f'{lat_lng}_satemb_vgg.pt')

        street_img = Image.open(street_img_name)

        satellite_img = Image.open(satellite_img_name)
        satellite_img = satellite_img.convert('RGB')

        satellite_emb = torch.load(satellite_emb_name, map_location='cpu').detach()
        satellite_emb_vgg = torch.load(satellite_emb_vgg_name, map_location='cpu').permute(2,0,1).detach()
        lat_emb, lng_emb = self.pos_enc(float(lat), float(lng))

        # cropping satellite image
        SAT_IMG_OUT_SIZE = 128
        left = (satellite_img.width - SAT_IMG_OUT_SIZE) / 2
        top = (satellite_img.height - SAT_IMG_OUT_SIZE) / 2
        right = (satellite_img.width + SAT_IMG_OUT_SIZE) / 2
        bottom = (satellite_img.height + SAT_IMG_OUT_SIZE) / 2

        if self.transform:
            satellite_img = self.transform(satellite_img)

        satellite_img = satellite_img.crop((left, top, right, bottom))

        if self.resize:
            street_img = self.resize(street_img)

        street_img = np.array(street_img).astype(np.uint8)
        street_img = (street_img / 127.5 - 1.0).astype(np.float32)

        satellite_img = np.array(satellite_img).astype(np.float32)
        satellite_img = (satellite_img / 127.5 - 1.0).astype(np.float32)

        sample = {'lat': lat, 'lng': lng, 'lat_emb': lat_emb, 'lng_emb': lng_emb, 
                  'street_image': street_img, 'satellite_image': satellite_img, 
                  'sat_emb': satellite_emb, 'sat_emb_vgg': satellite_emb_vgg}

        return sample
    
    def pos_enc(self, lat, lng, model_dim=128):
        lat, lng = lat, lng
        
        MIN_LAT = 24.396308     # key west, florida
        MAX_LAT = 49.384358     # northwest angle, minnesota 
        MIN_LON = -125.000000   # cape alava, washington
        MAX_LON = -66.934570    # west quoddy head, maine

        norm_lat = (lat / ((MAX_LAT - MIN_LAT) / 2) - 1.0)
        norm_lng = (lng / ((MAX_LON - MIN_LON) / 2) - 1.0)

        inds = torch.arange(model_dim)
        
        enc_lng = torch.sin(norm_lng / (10000 ** (2 * inds / model_dim)))
        enc_lat = torch.sin(norm_lat / (10000 ** (2 * inds / model_dim)))

        return enc_lat, enc_lng



    
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

if __name__ == '__main__':
    data = DataLoader(StreetSatTrain(), batch_size=16)
    for idx, batch in enumerate(tqdm(data)):
        sat_emb_vgg = batch["sat_emb"]
        if sat_emb_vgg.shape[0] < 8:
            print('heree!')
            sat_emb_vgg = batch["sat_emb"]
    