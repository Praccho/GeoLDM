import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# CAREFUL RUNNING THIS

if __name__ == '__main__':
    type = 'street' # 'street' or 'sat'
    src_dir = f'raw/{type}'
    to_dir = f'raw/{type}'
    # down_size = 64 # for street
    # down_size = 192 # for sat
    resize = transforms.Resize(down_size)
    for filename in tqdm(os.listdir(dir)):
        if not filename.endswith(".png") or filename.startswith('.'):
            continue
        filepath = os.path.join(dir, filename)
        with Image.open(filepath) as img:
            img_resized = resize(img)
            img_resized.save(filepath)