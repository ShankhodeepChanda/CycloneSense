from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path("kaggle_multiyear45")
INDEX = ROOT / "images_index.csv"

if not INDEX.exists():
    raise FileNotFoundError(INDEX)

df = pd.read_csv(INDEX)
print("INDEX COLUMNS:", df.columns.tolist())
print("TOTAL ROWS:", len(df))

required = ["sid", "time", "wind_kt", "channel", "split", "image_path"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")
df["time_parsed"] = pd.to_datetime(df["time"], errors="coerce")

print("\nROWS BY CHANNEL AND SPLIT")
print(pd.crosstab(df["channel"], df["split"]).to_string())

print("\nMISSING VALUES")
print(df[required].isna().sum().to_string())

print("\nDUPLICATE IMAGE PATHS:", int(df["image_path"].duplicated().sum()))
print("DUPLICATE CHANNEL/STORM/TIME:")
print(int(df.duplicated(["channel", "sid", "time"]).sum()))

print("\nIMAGE FILE CHECK")
missing_files = []
unreadable = []
bad_dimensions = []

for _, row in df.iterrows():
    path = ROOT / str(row["image_path"])
    if not path.exists():
        missing_files.append(str(path))
        continue
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            if im.width < 32 or im.height < 32:
                bad_dimensions.append((str(path), im.size))
    except Exception as exc:
        unreadable.append((str(path), str(exc)))

print("Missing files:", len(missing_files))
print("Unreadable files:", len(unreadable))
print("Very small images:", len(bad_dimensions))

if missing_files:
    print("Examples of missing files:", missing_files[:5])
if unreadable:
    print("Examples of unreadable files:", unreadable[:5])
if bad_dimensions:
    print("Examples of small images:", bad_dimensions[:5])

print("\nWIND LABEL SUMMARY BY CHANNEL")
print(df.groupby("channel")["wind_kt"].describe().round(2).to_string())

print("\nSTORM SPLIT OVERLAP CHECK")
split_storms = {
    split: set(part["sid"].dropna().astype(str))
    for split, part in df.groupby("split")
}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    print(f"{a} ∩ {b}:", len(split_storms.get(a, set()) & split_storms.get(b, set())))

print("\nAUDIT COMPLETE — no files were modified.")
