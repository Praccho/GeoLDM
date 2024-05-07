import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import FrechetInceptionDistance as FID

class Metrics(nn.Module):
    def __init__(self, gen_imgs, street_imgs):
        self.gen_imgs, self.street_imgs = gen_imgs, street_imgs

        self.ssim = SSIM(gen_imgs, street_imgs, reduction="none")

        self.fid = FID(device=torch.device('cuda'))
        self.fid.update(gen_imgs, real=False)
        self.fid.update(street_imgs, real=True)

    def get_metrics(self):
        return {'ssim': torch.mean(self.ssim), 'fid': self.fid.compute()}
    
    def top_results(self, gen_imgs, n=5):
        ssim_inds = torch.argsort(self.ssim)[:n]
        ssim_gen = torch.index_select(self.gen_imgs, 0, ssim_inds)
        ssim_street = torch.index_select(self.street_imgs, 0, ssim_inds)

        fid_inds = torch.argsort(self.fid, descending=True)[:n]
        fid_gen = torch.index_select(self.gen_imgs, 0, fid_inds)
        fid_street = torch.index_select(self.street_imgs, 0, fid_inds)

        return {'ssim': zip(ssim_gen, ssim_street), 'fid': zip(fid_gen, fid_street)}


#-----------------------------------------------------------------------------

if __name__ == '__main__':
    # add in gen_imgs and street_imgs 

    metrics = Metrics(_, _)