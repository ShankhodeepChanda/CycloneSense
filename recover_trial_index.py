from pathlib import Path
import pandas as pd
from data.build_dataset import load_tracks, plan_chips, SATELLITES, CHANNELS

out = Path("kaggle_trial")
tracks = pd.read_csv(out / "tracks.csv", parse_dates=["time"])
jobs = plan_chips(tracks, 10.0)
rows = []

for job in jobs:
    for channel in CHANNELS:
        rel = Path("images") / channel / job["sat"] / f"{job['sid']}_{job['date'].replace('-', '')}.jpg"
        path = out / rel
        if not path.is_file():
            continue
        layer = SATELLITES[job["sat"]][2 if channel == "truecolor" else 3]
        rows.append({
            "image_path": rel.as_posix(),
            "channel": channel,
            "satellite": job["sat"],
            "layer": layer,
            "chip_deg": 10.0,
            **{k: job[k] for k in ("sid", "name", "basin", "time", "lat", "lon",
                                   "wind_kt", "pres_hpa", "category", "ri_24h",
                                   "split", "offset_h")}
        })

index = pd.DataFrame(rows)
if not index.empty:
    index = index.drop_duplicates(subset=["image_path"])
    index = index.sort_values(["sid", "time", "channel", "satellite"])

index.to_csv(out / "images_index.csv", index=False)
print("Recovered index rows:", len(index))
print("Channels:", index["channel"].value_counts().to_dict() if len(index) else {})
print("Storms per split:")
print(index.groupby("split")["sid"].nunique() if len(index) else "No indexed images")
print("All indexed paths exist:",
      all((out / p).is_file() for p in index["image_path"]))
