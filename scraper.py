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
import data_cleaner

def generate_random_us_coordinates():
    while True:
        latitude = np.random.uniform(24.5, 49.0)
        longitude = np.random.uniform(-125.0, -67.0)
        if data_cleaner.is_point_in_us_mainland(latitude, longitude):
            continue
        return latitude, longitude


if __name__ == "__main__":
    IMG_SIZE = 400
    STRT_IMG_OUT_SIZE = 256
    SAT_IMG_OUT_SIZE = 384

    API_ENDPOINT = 'https://maps.googleapis.com/maps/api/streetview'

    load_dotenv()
    API_KEY = os.getenv('API_KEY')

    # number of images to be obtained
    NUM_IMAGES = 10000
    MAX_QUERIES = 100000
    RADIUS = 25000
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=chrome_options)

    hits = 0

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
            if "location" not in img_data:
                continue
            lat, lng = img_data["location"]["lat"], img_data["location"]["lng"]
            loc = f'{lat},{lng}'

            rand_heading = np.random.choice(np.arange(365))

            api_request = f'{API_ENDPOINT}?size={IMG_SIZE}x{IMG_SIZE}&location={loc}&fov=80&heading={rand_heading}&pitch=0&key={API_KEY}'
            driver.get(api_request)
            
            street_out_path = f'data/train/street/{loc}_street.png'
            sat_out_path = f'data/train/satellite/{loc}_sat.png'

            if os.path.exists(street_out_path):
                continue

            try:
                image_url = driver.find_element(By.TAG_NAME, 'img').get_attribute('src')
                response = requests.get(image_url)
                street_img = Image.open(BytesIO(response.content))
                left = (street_img.width - STRT_IMG_OUT_SIZE) / 2
                top = (street_img.height - STRT_IMG_OUT_SIZE) / 2
                right = (street_img.width + STRT_IMG_OUT_SIZE) / 2
                bottom = (street_img.height + STRT_IMG_OUT_SIZE) / 2
            
                street_img_cropped = street_img.crop((left, top, right, bottom))
            except:
                continue

            sat_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=19&size=600x600&maptype=satellite&key={API_KEY}"

            try:
                response = requests.get(sat_url)
                sat_img = Image.open(BytesIO(response.content))

                left = (sat_img.width - SAT_IMG_OUT_SIZE) / 2
                top = (sat_img.height - SAT_IMG_OUT_SIZE) / 2
                right = (sat_img.width + SAT_IMG_OUT_SIZE) / 2
                bottom = (sat_img.height + SAT_IMG_OUT_SIZE) / 2

                sat_img_cropped = sat_img.crop((left, top, right, bottom))
            except:
                continue     
            street_img_cropped.save(street_out_path)
            sat_img_cropped.save(sat_out_path)
