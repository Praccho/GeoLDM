import requests
from PIL import Image
from io import BytesIO

# Replace these with your actual details
API_KEY = ''
latitude = 40.7128
longitude = -74.0060
IMG_OUT_SIZE = 300  # The size of the cropped image

def fetch_and_crop_image(lat, lng, api_key, img_out_size, out_path):
    # Construct the Google Maps Static API URL
    img_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=12&size=600x600&maptype=satellite&key={api_key}"

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

# Example usage
output_path = "C:\\Users\\seoli\\GeoLDM\\data\\satellite\\map_image.png"
fetch_and_crop_image(latitude, longitude, API_KEY, IMG_OUT_SIZE, output_path)
