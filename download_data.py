import os
import urllib.request
import urllib.parse
import numpy as np

CATEGORIES = [
    # Animals
    "cat", "dog", "fish", "bird", "elephant", "horse", "rabbit", "snake",
    "owl", "penguin", "bear", "frog", "duck", "shark", "whale", "octopus",
    "crab", "sea turtle", "butterfly", "bee",
    # Vehicles
    "car", "bicycle", "airplane", "sailboat", "train", "helicopter", "bus", "motorbike",
    # Food
    "pizza", "apple", "banana", "grapes", "strawberry", "ice cream", "donut",
    "mushroom", "cookie", "watermelon",
    # Nature
    "tree", "flower", "sun", "moon", "star", "cloud", "mountain", "cactus",
    # Objects
    "house", "clock", "umbrella", "key", "eyeglasses", "hat", "shoe", "cup",
    "book", "guitar", "crown", "candle", "campfire", "diamond",
    # Symbols
    "octagon", "smiley face", "snowman", "eye", "face",
    # Added
    "lion", "tiger", "panda", "camera",
]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
SAVE_DIR = "data"
MAX_SAMPLES = 10000

os.makedirs(SAVE_DIR, exist_ok=True)

for i, category in enumerate(CATEGORIES, 1):
    filename = category + ".npy"
    url = BASE_URL + urllib.parse.quote(filename)
    save_path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(save_path):
        print(f"[{i}/{len(CATEGORIES)}] Already exists: {category}")
        continue

    print(f"[{i}/{len(CATEGORIES)}] Downloading: {category}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, save_path)
        data = np.load(save_path)
        if len(data) > MAX_SAMPLES:
            np.save(save_path, data[:MAX_SAMPLES])
        print(f"done ({min(len(data), MAX_SAMPLES)} samples)")
    except Exception as e:
        print(f"FAILED — {e}")

print("\nAll done.")
