import os
import requests
from bs4 import BeautifulSoup

URL = 'https://www.google.com/search?sca_esv=575507320&hl=en-US&sxsrf=AM9HkKnRBz2WPMFLOcoMLeg-Lu1egX0rJw:1697930614994&q=Whale&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjxtKO5pIiCAxXNOEQIHd2BCiIQ0pQJegQIChAB&biw=1219&bih=855&dpr=2'
SAVE_DIR = 'downloaded_whales'

# Ensure the save directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

response = requests.get(URL)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all image tags
img_tags = soup.find_all('img')

for img_tag in img_tags:
    img_url = img_tag['src']
    alt_text = img_tag.get('alt', '').lower()

    # Check if the image filename or alt text contains the word 'whale'
    if 'whale' in img_url.lower() or 'whale' in alt_text:
        img_data = requests.get(img_url).content
        img_filename = os.path.join(SAVE_DIR, os.path.basename(img_url))

        with open(img_filename, 'wb') as img_file:
            img_file.write(img_data)
        print(f"Downloaded {img_url} to {img_filename}")

print("Download complete!")
