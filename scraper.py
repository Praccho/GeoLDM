import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import pandas as pd
import numpy as np
import json
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

IMG_SIZE = 384
IMG_OUT_SIZE = 256
API_ENDPOINT = 'https://maps.googleapis.com/maps/api/streetview'

load_dotenv()
API_KEY = os.getenv('API_KEY')

# number of images to be obtained
NUM_IMAGES = 1000
MAX_QUERIES = 10000
RADIUS = 25000

def generate_random_us_coordinates():
    # Latitude for the mainland US approximately ranges from 24.5 to 49
    # Longitude for the mainland US approximately ranges from -125 to -67
    latitude = np.random.uniform(24.5, 49.0)
    longitude = np.random.uniform(-125.0, -67.0)
    return latitude, longitude

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

# set this to the path of the driver you download @ https://chromedriver.storage.googleapis.com/index.html
# os.environ['PATH'] += r"/Documents/CS1430_Projects/chromedriver_mac_arm64/chromedriver"
driver = webdriver.Chrome(options=chrome_options)

hits = 0
seen = 0
total = 0

for i in range(MAX_QUERIES):
    if hits == NUM_IMAGES:
        break

    rand_lat, rand_lng = generate_random_us_coordinates()
    rand_loc = f'{rand_lat},{rand_lng}'

    metadata_request = f'{API_ENDPOINT}/metadata?&location={rand_loc}&radius={RADIUS}&key={API_KEY}'
    driver.get(metadata_request)
    
    try:
        text_element = driver.find_element(By.XPATH, "//*[contains(text(), 'ZERO_RESULTS')]")
    except:
        img_json_text = driver.find_element(by="tag name", value="body").text
        img_data = json.loads(img_json_text)
        lat, lng = img_data["location"]["lat"], img_data["location"]["lng"]
        loc = f'{lat},{lng}'

        rand_heading = np.random.choice(np.arange(365))

        api_request = f'{API_ENDPOINT}?size={IMG_SIZE}x{IMG_SIZE}&location={loc}&fov=80&heading={rand_heading}&pitch=0&key={API_KEY}'
        driver.get(api_request)
        
        out_path = f'data/street/{loc}_street.png'

        if os.path.exists(out_path):
            seen += 1
            continue

        try:
            image_url = driver.find_element(By.TAG_NAME, 'img').get_attribute('src')
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            left = (img.width - IMG_OUT_SIZE) / 2
            top = (img.height - IMG_OUT_SIZE) / 2
            right = (img.width + IMG_OUT_SIZE) / 2
            bottom = (img.height + IMG_OUT_SIZE) / 2
        
            img_cropped = img.crop((left, top, right, bottom))
            img_cropped.save(out_path)
            hits += 1
        except:
            pass
    total += 1

print("seen:", seen)
print(f"hit rate: {hits}/{total} =", hits / total)