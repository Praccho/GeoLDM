import os

if __name__ == '__main__':
    sat_dir = 'data/train/satemb_vgg'
    emb_dir = 'data/train/satemb_vgg'

    for file in os.listdir(sat_dir):
        if os.path.splitext(file)[1] != '.pt':
            pass
        coords = str.split(file, '_')[0]
        if not os.path.exists(os.path.join(emb_dir, coords + '_satemb_vgg.pt')):
            print('DNE', os.path.join(emb_dir, coords + '_satemb_vgg.pt'))