import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# CAREFUL RUNNING THIS

if __name__ == '__main__':

    for type, down_size in [('street', 64), ('satellite', 192)]:
        resize = transforms.Resize(down_size)
        src_dir = f'raw/{type}'
        to_dir = f'train/{type}'
        for filename in tqdm(os.listdir(src_dir)):
            if not filename.endswith(".png") or filename.startswith('.'):
                continue
            filepath = os.path.join(src_dir, filename)
            outpath = os.path.join(to_dir, filename)
            with Image.open(filepath) as img:
                img_resized = resize(img)
                img_resized.save(outpath)