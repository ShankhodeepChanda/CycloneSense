from pathlib import Path
import pandas as pd
from PIL import Image

root = Path("kaggle_dataset")
df = pd.read_csv(root / "images_index.csv")
bad = []
sizes = {}

for rel in df["image_path"].drop_duplicates():
    path = root / rel
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
    except Exception as e:
        bad.append((str(rel), str(e)))

print("Indexed rows:", len(df))
print("Unique image paths:", df["image_path"].nunique())
print("Unreadable images:", len(bad))
print("Image dimensions:", sizes)
print("Examples of bad files:", bad[:10])
