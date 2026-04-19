import os
import requests

# random images (no cars)
URLS = [
    "https://picsum.photos/400/300",
    "https://picsum.photos/500/400",
    "https://picsum.photos/600/400"
]

SAVE_DIR = "data/negatives"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_images(n=500):
    for i in range(n):
        url = URLS[i % len(URLS)]
        try:
            img_data = requests.get(url, timeout=5).content
            with open(f"{SAVE_DIR}/neg_{i}.jpg", "wb") as f:
                f.write(img_data)
        except:
            continue

if __name__ == "__main__":
    download_images(500)