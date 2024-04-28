import geopandas as gpd
from shapely.geometry import Point
import os
from PIL import Image
from tqdm import tqdm

gdf = gpd.read_file('assets/cb_2018_us_state_500k.shp')


def is_point_in_us_mainland(lat, lng):
    point = Point(lng, lat)
    return any(gdf.contains(point))

def remove_non_mainland_image_pairs(street_dir, satellite_dir):
    for filename in tqdm(os.listdir(street_dir)):
        if not filename.endswith(".png") or filename.startswith('.'):
            continue

        base_name, _ = os.path.splitext(filename)
        coords, _ = base_name.split('_')
        lat, lng = map(float, coords.split(','))

        if not is_point_in_us_mainland(lat, lng):
            street_image_path = os.path.join(street_dir, filename)
            satellite_image_path = os.path.join(satellite_dir, f"{coords}_sat.png")

            if os.path.exists(street_image_path) and os.path.exists(satellite_image_path):
                os.remove(street_image_path)
                os.remove(satellite_image_path)

def remove_missing(street_dir, satellite_dir):
    bad_sample_dir = 'data/bad_samples'
    bad_images = []
    for filename in os.listdir(bad_sample_dir):
        if filename.endswith("png"):
            bad_images.append(list(Image.open(os.path.join(bad_sample_dir, filename)).getdata()))
    
    for filename in tqdm(os.listdir(street_dir)):
        if not filename.endswith(".png") or filename.startswith('.'):
            continue

        base_name, _ = os.path.splitext(filename)
        coords, _ = base_name.split('_')

        street_image_path = os.path.join(street_dir, filename)
        satellite_image_path = os.path.join(satellite_dir, f"{coords}_sat.png")

        street_image = list(Image.open(street_image_path).getdata())
        satellite_image = list(Image.open(satellite_image_path).getdata())

        for bad_img in bad_images:
            if street_image == bad_img or satellite_image == bad_img:
                os.remove(street_image_path)
                os.remove(satellite_image_path)

if __name__ == "__main__":
    street_dir = 'data/val/street'
    satellite_dir = 'data/val/satellite'

    # remove_non_mainland_image_pairs(street_dir, satellite_dir)
    remove_missing(street_dir, satellite_dir)
