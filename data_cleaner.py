import geopandas as gpd
from shapely.geometry import Point
import os

gdf = gpd.read_file('assets/cb_2018_us_state_500k.shp')

def is_point_in_us_mainland(lat, lng):
    point = Point(lng, lat)
    return any(gdf.contains(point))

def remove_non_mainland_image_pairs(street_dir, satellite_dir):
    for filename in os.listdir(street_dir):
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

street_dir = 'data/street'
satellite_dir = 'data/satellite'

remove_non_mainland_image_pairs(street_dir, satellite_dir)