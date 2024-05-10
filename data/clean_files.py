import os
import tqdm as tqdm
import math

if __name__ == '__main__':
    src_dir = f'test/satellite'
    to_dir = f'test/satemb_vgg'

    satellite_filenames = sorted(os.listdir(src_dir))
    satemb_filenames = sorted(os.listdir(to_dir))
    unmatched_sat = []
    unmatched_satemb = []

    for (sat_fn, satemb_fn) in zip(satellite_filenames, satemb_filenames):
        if not sat_fn.endswith(".png") or sat_fn.startswith('.'):
                continue
        
        sat_lat_lng = sat_fn.split('_')[0]
        sat_lat, sat_lng = sat_lat_lng.split(',')

        emb_lat_lng = satemb_fn.split('_')[0]
        emb_lat, emb_lng = emb_lat_lng.split(',')

        if math.isclose(float(sat_lat), float(emb_lat), rel_tol=0.00005) and math.isclose(float(sat_lng), float(emb_lng), rel_tol=0.00005):
            filepath = os.path.join(to_dir, satemb_fn)
            outpath = os.path.join(to_dir, f'{sat_lat_lng}_satemb_vgg.pt')
            os.rename(filepath, outpath)
        else:
            unmatched_sat.append(sat_fn)
            unmatched_satemb.append((emb_lat_lng, satemb_fn))


    for sat_fn in unmatched_sat:
        if not sat_fn.endswith(".png") or sat_fn.startswith('.'):
                continue
        
        sat_lat_lng = sat_fn.split('_')[0]
        sat_lat, sat_lng = sat_lat_lng.split(',')
        
        for (emb_lat_lng, satemb_fn) in unmatched_satemb:
            emb_lat, emb_lng = emb_lat_lng.split(',')
            if sat_lat_lng == emb_lat_lng: break
            if math.isclose(float(sat_lat), float(emb_lat), rel_tol=0.00005) and math.isclose(float(sat_lng), float(emb_lng), rel_tol=0.00005):
                filepath = os.path.join(to_dir, satemb_fn)
                outpath = os.path.join(to_dir, f'{sat_lat_lng}_satemb_vgg.pt')
                os.rename(filepath, outpath)
                break


        print("Satemb:", satemb_fn)
        print("Sat: ", sat_fn)