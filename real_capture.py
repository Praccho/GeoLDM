import requests
from PIL import Image
from io import BytesIO
import json 
import os
from dotenv import load_dotenv
import random

load_dotenv()
API_KEY = os.getenv('API_KEY')

# Replace these with your actual details
IMG_OUT_SIZE = 300  # The size of the cropped image

def fetch_and_crop_image(lat, lng, api_key, img_out_size, out_path):
    # Construct the Google Maps Static API URL
    img_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=19&size=600x600&maptype=satellite&key={api_key}"

    try:
        # Fetch the image
        response = requests.get(img_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))

            left = (img.width - img_out_size) / 2
            top = (img.height - img_out_size) / 2
            right = (img.width + img_out_size) / 2
            bottom = (img.height + img_out_size) / 2

            # Crop and save the image
            img_cropped = img.crop((left, top, right, bottom))
            img_cropped.save(out_path)
            print(f"Image successfully saved to {out_path}")
        else:
            print(f"Failed to fetch image, status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

with open('coordinates.json', 'r') as f:
    coordinates = json.load(f)


satellite_dir = "data\satellite"
os.makedirs(satellite_dir, exist_ok=True)

for coord in coordinates:
    lat = coord['lat']
    lng = coord['lng']
    output_path = os.path.join(satellite_dir, f"{lat},{lng}_satellite.png")
    fetch_and_crop_image(lat, lng, API_KEY, IMG_OUT_SIZE, output_path)