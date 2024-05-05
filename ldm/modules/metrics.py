import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import FrechetInceptionDistance as FID

class Metrics(nn.Module):
    def __init__(self, gen_imgs, street_imgs):
        self.ssim = SSIM(gen_imgs, street_imgs)
        self.fid = FID(device=torch.device('cuda'))
        