import gif
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

@gif.frame
def plot(img):
    plt.axis()
    plt.imshow(img)

if __name__ == '__main__':
    src_dir = 'vae_samples'
    rate = 8
    filenames = [file for file in os.listdir(src_dir) if file[-3:]=='png']
    filenames.sort()
    count = (len(filenames) + rate) // rate
    width, height = Image.open(os.path.join(src_dir, filenames[0])).size
    frames = np.zeros((count, height, width, 3)).astype(int)
    for i, file in enumerate(filenames):

        if i % rate == 0:
            frame = np.array(Image.open(os.path.join(src_dir, file)))
            frames[i // rate] = frame.astype(int)

    frames_gif = [plot(frame) for frame in frames]

    gif.save(frames_gif, 'vae.gif', duration=200)


