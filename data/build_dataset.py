"""Build a real, large cyclone dataset and package it for Kaggle.

Labels come from NOAA IBTrACS best-track data (every basin). Images are real
NASA GIBS MODIS (Terra + Aqua) chips centred on each storm: true colour and
Band-31 brightness temperature (IR, palette-rendered), fetched with no API key.

Nothing here is simulated. Run it on a machine with internet access:

    python data/build_dataset.py --check                      # 1 request per layer
    python data/build_dataset.py --limit 50                   # small trial run
    python data/build_dataset.py --start-year 2003 --workers 8 --kaggle-user YOU

Output (default ./kaggle_dataset):
    tracks.csv            every 6-hourly best-track record with labels + split
    images_index.csv      one row per saved image chip, with labels + split
    images/<channel>/<satellite>/<storm>_<date>.jpg
    dataset-metadata.json (Kaggle)

Re-running resumes: chips already on disk are reused, not downloaded again.
Publish:  kaggle datasets create -p kaggle_dataset -r zip
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
)
GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

# satellite -> (local solar hour of the daytime pass, first date with imagery, true-colour layer, IR layer)
SATELLITES = {
    "terra": (10.5, "2000-03-01", "MODIS_Terra_CorrectedReflectance_TrueColor", "MODIS_Terra_Brightness_Temp_Band31_Day"),
    "aqua": (13.5, "2002-07-05", "MODIS_Aqua_CorrectedReflectance_TrueColor", "MODIS_Aqua_Brightness_Temp_Band31_Day"),
}
CHANNELS = ("truecolor", "ir")

WANTED = {
    "SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "TRACK_TYPE", "LAT", "LON",
    "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES", "DIST2LAND",
}
NUMERIC = ["LAT", "LON", "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES", "DIST2LAND"]


# ----------------------------------------------------------------- labels

def category(kt: float) -> str:
    """Saffir-Simpson style class from 1-minute sustained wind in knots."""
    for limit, name in ((34, "TD"), (64, "TS"), (83, "C1"), (96, "C2"), (113, "C3"), (137, "C4")):
        if kt < limit:
            return name
    return "C5"


def split_of(sid: str) -> str:
    """Deterministic 70/15/15 split by storm ID so no storm spans two splits."""
    bucket = int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def load_tracks(csv_path: str | Path, start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=[1], usecols=lambda c: c in WANTED, dtype=str, low_memory=False)
    missing = WANTED - set(df.columns)
    if missing:
        sys.exit(f"IBTrACS file is missing expected columns {sorted(missing)}; format may have changed.")

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df[(df["TRACK_TYPE"] == "main") & df["time"].notna()]
    df = df[df["time"].dt.year.between(start_year, end_year)]
    # Keep the 6-hourly synoptic times only (IBTrACS also has 3-hourly interpolations).
    df = df[(df["time"].dt.hour % 6 == 0) & (df["time"].dt.minute == 0)]

    # One consistent wind definition across basins: USA 1-minute sustained wind (kt).
    df = df[df["USA_WIND"] > 0].drop_duplicates(["SID", "time"]).copy()
    df = df.rename(columns={"USA_WIND": "wind_kt", "USA_PRES": "pres_hpa", "LAT": "lat", "LON": "lon"})
    df["category"] = df["wind_kt"].map(category)
    df = df.sort_values(["SID", "time"]).reset_index(drop=True)

    def shifted(hours: int, cols: dict[str, str]) -> pd.DataFrame:
        other = df[["SID", "time", *cols]].rename(columns=cols)
        other["time"] = other["time"] - pd.Timedelta(hours=hours)  # value from +hours ahead lands on this row
        return other

    # Previous 6 h (for deltas) and +24 h (for the rapid-intensification label).
    prev = df[["SID", "time", "lat", "lon", "wind_kt"]].rename(
        columns={"lat": "lat_prev", "lon": "lon_prev", "wind_kt": "wind_prev"}
    )
    prev["time"] = prev["time"] + pd.Timedelta(hours=6)
    df = df.merge(prev, on=["SID", "time"], how="left")
    df = df.merge(shifted(24, {"wind_kt": "wind_24h"}), on=["SID", "time"], how="left")

    df["dlat"] = df["lat"] - df["lat_prev"]
    df["dlon"] = df["lon"] - df["lon_prev"]
    df["dwind"] = df["wind_kt"] - df["wind_prev"]
    df["dwind_24h"] = df["wind_24h"] - df["wind_kt"]
    # Rapid intensification: wind gain >= 30 kt in 24 h (standard definition). NaN when unknown.
    df["ri_24h"] = np.where(df["wind_24h"].isna(), np.nan, (df["dwind_24h"] >= 30).astype(float))
    df["split"] = df["SID"].map(split_of)

    keep = ["SID", "NAME", "SEASON", "BASIN", "time", "lat", "lon", "wind_kt", "pres_hpa", "WMO_WIND", "WMO_PRES",
            "DIST2LAND", "category", "dlat", "dlon", "dwind", "dwind_24h", "ri_24h", "split"]
    return df[keep].rename(columns={"SID": "sid", "NAME": "name", "SEASON": "season", "BASIN": "basin",
                                    "WMO_WIND": "wmo_wind_kt", "WMO_PRES": "wmo_pres_hpa", "DIST2LAND": "dist2land_km"})


# ------------------------------------------------------------------ chips

def plan_chips(tracks: pd.DataFrame, chip_deg: float) -> list[dict]:
    """For each satellite, pick per storm-day the record nearest the satellite's daytime overpass."""
    jobs: list[dict] = []
    half = chip_deg / 2
    for sat, (pass_hour, first_date, *_layers) in SATELLITES.items():
        t = tracks[(tracks["lat"].abs() + half < 90) & (tracks["lon"].abs() + half < 180)].copy()
        local = (t["time"].dt.hour + t["time"].dt.minute / 60 + t["lon"] / 15) % 24
        t["offset_h"] = ((local - pass_hour + 12) % 24) - 12  # hours between record and overpass
        t["pass_date"] = (t["time"] - pd.to_timedelta(t["offset_h"], unit="h")).dt.strftime("%Y-%m-%d")
        t = t[(t["pass_date"] >= first_date) & (t["offset_h"].abs() <= 3.0)]
        t = t.loc[t.assign(a=t["offset_h"].abs()).groupby(["sid", "pass_date"])["a"].idxmin()]
        for r in t.itertuples():
            jobs.append({"sat": sat, "date": r.pass_date, "sid": r.sid, "name": r.name, "basin": r.basin,
                         "time": r.time, "lat": r.lat, "lon": r.lon, "wind_kt": r.wind_kt,
                         "pres_hpa": r.pres_hpa, "category": r.category, "ri_24h": r.ri_24h,
                         "split": r.split, "offset_h": round(float(r.offset_h), 2)})
    return jobs


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    return s


def fetch_chip(session: requests.Session, layer: str, date: str, lat: float, lon: float,
               chip_deg: float, size: int) -> tuple[Image.Image, float] | None:
    """Return (RGB image, fraction of pixels with data) or None if GIBS had no image."""
    h = chip_deg / 2
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer,
        "CRS": "EPSG:4326", "BBOX": f"{lat - h},{lon - h},{lat + h},{lon + h}",  # 1.3.0 + 4326 = lat,lon order
        "WIDTH": size, "HEIGHT": size, "TIME": date,
        "FORMAT": "image/png" if "Brightness_Temp" in layer else "image/jpeg",
    }
    r = session.get(GIBS_WMS, params=params, timeout=60)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
        return None
    return decode_chip(r.content)


def decode_chip(raw: bytes) -> tuple[Image.Image, float]:
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        valid = float((np.asarray(rgba)[..., 3] > 0).mean())  # transparent = no data
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
    else:
        valid = float((np.asarray(img.convert("RGB")).max(axis=2) > 8).mean())  # black = no data
    return img.convert("RGB"), valid


def do_job(session: requests.Session, job: dict, channel: str, out: Path, chip_deg: float,
           size: int, min_valid: float) -> dict | None:
    layer = SATELLITES[job["sat"]][2 if channel == "truecolor" else 3]
    rel = Path("images") / channel / job["sat"] / f"{job['sid']}_{job['date'].replace('-', '')}.jpg"
    path = out / rel
    try:
        if path.exists():
            with Image.open(path) as im:
                valid = float((np.asarray(im.convert("RGB")).max(axis=2) > 8).mean())
        else:
            got = fetch_chip(session, layer, job["date"], job["lat"], job["lon"], chip_deg, size)
            if got is None:
                return None
            img, valid = got
            if valid < min_valid:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, quality=90)
    except (requests.RequestException, OSError):
        return None
    if valid < min_valid:
        return None
    return {"image_path": rel.as_posix(), "channel": channel, "satellite": job["sat"], "layer": layer,
            "valid_fraction": round(valid, 3), "chip_deg": chip_deg,
            **{k: job[k] for k in ("sid", "name", "basin", "time", "lat", "lon", "wind_kt", "pres_hpa",
                                   "category", "ri_24h", "split", "offset_h")}}


def check_layers() -> None:
    """One real request per layer so a wrong layer name or blocked network shows up in seconds."""
    session = make_session()
    date = "2024-05-26"  # Cyclone Remal (Bay of Bengal)
    for sat, (_h, _first, tc, ir) in SATELLITES.items():
        for layer in (tc, ir):
            try:
                got = fetch_chip(session, layer, date, 19.0, 89.0, 10.0, 256)
                print(f"{layer}: {'OK, valid=%.2f' % got[1] if got else 'NO IMAGE (check layer name/date)'}")
            except requests.RequestException as exc:
                print(f"{layer}: NETWORK ERROR {exc}")


# ----------------------------------------------------------------- Kaggle

def write_metadata(out: Path, user: str, tracks: pd.DataFrame, index: pd.DataFrame) -> None:
    desc = (
        "Real tropical-cyclone dataset. Labels: NOAA IBTrACS v04r01 best-track (USA 1-minute sustained wind, "
        "pressure, position, all basins). Images: NASA GIBS MODIS Terra/Aqua chips centred on the storm "
        f"({index['chip_deg'].iloc[0] if len(index) else 10:g} degree box, true colour and Band-31 brightness "
        "temperature rendered with GIBS palette), nearest daytime overpass.\n\n"
        "Split by storm ID (70/15/15): never random by image. Files: tracks.csv (every 6-hourly record, with "
        "6 h deltas and 24 h rapid-intensification flag), images_index.csv (one row per image). "
        "`valid_fraction` is the share of pixels with data; `offset_h` is hours between the best-track time and "
        "the satellite pass. Limitations: one daytime pass per day (no night IR), storms near the dateline are "
        "skipped, wind agencies differ by basin so use `category`/`wind_kt` consistently. Seven Dvorak pattern "
        "labels are NOT included: IBTrACS has none.\n\n"
        "Sources / citation: Knapp et al. 2010 (IBTrACS), NOAA NCEI; NASA GIBS / MODIS (NASA EOSDIS). "
        "Check each source's current terms before redistribution."
    )
    meta = {
        "title": "CycloneSense: Satellite Chips + Best Track",  # Kaggle: 6-50 chars
        "id": f"{user}/cyclonesense-tc-satellite",
        "subtitle": "Real MODIS chips and IBTrACS labels, split by storm",
        "description": desc,
        "licenses": [{"name": "other"}],
        "keywords": ["weather", "satellite imagery", "cyclone", "remote sensing", "deep learning"],
    }
    assert 6 <= len(meta["title"]) <= 50 and 20 <= len(meta["subtitle"]) <= 80, "Kaggle title/subtitle length"
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


# ------------------------------------------------------------------- main

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="kaggle_dataset")
    p.add_argument("--cache", default="data/raw", help="where the IBTrACS CSV is cached")
    p.add_argument("--start-year", type=int, default=2003)
    p.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year - 1)
    p.add_argument("--chip-deg", type=float, default=10.0, help="chip width/height in degrees")
    p.add_argument("--size", type=int, default=256, help="chip size in pixels")
    p.add_argument("--min-valid", type=float, default=0.6, help="drop chips with less data than this")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="only fetch the first N storm-days (trial run)")
    p.add_argument("--tracks-only", action="store_true")
    p.add_argument("--check", action="store_true", help="test every GIBS layer and exit")
    p.add_argument("--kaggle-user", default="YOUR_KAGGLE_USERNAME")
    args = p.parse_args()

    if args.check:
        return check_layers()

    csv_path = Path(args.cache) / "ibtracs.since1980.list.v04r01.csv"
    if not csv_path.exists():
        download(IBTRACS_URL, csv_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(csv_path, args.start_year, args.end_year)
    tracks.to_csv(out / "tracks.csv", index=False)
    print(f"tracks.csv: {len(tracks)} records, {tracks['sid'].nunique()} storms")

    index = pd.DataFrame()
    if not args.tracks_only:
        jobs = plan_chips(tracks, args.chip_deg)
        if args.limit:
            jobs = jobs[: args.limit]
        print(f"{len(jobs)} storm-days x {len(CHANNELS)} channels to fetch")
        session = make_session()
        rows, done, failed = [], 0, 0
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(do_job, session, j, ch, out, args.chip_deg, args.size, args.min_valid)
                       for j in jobs for ch in CHANNELS]
            for fut in as_completed(futures):
                row = fut.result()
                done += 1
                failed += row is None
                if row:
                    rows.append(row)
                if done % 500 == 0:
                    print(f"  {done}/{len(futures)} done, {len(rows)} kept")
                if done == 40 and failed == 40:
                    pool.shutdown(cancel_futures=True)
                    sys.exit("First 40 requests all failed: run with --check to see whether it is the network or a layer name.")
        index = pd.DataFrame(rows).sort_values(["sid", "time", "channel", "satellite"]) if rows else pd.DataFrame()
        index.to_csv(out / "images_index.csv", index=False)
        print(f"images_index.csv: {len(index)} images kept, {failed} skipped (no data / low coverage)")
        if len(index):
            print(index.groupby(["split", "category"]).size().unstack(fill_value=0))

    write_metadata(out, args.kaggle_user, tracks, index)
    print(f"Done. Publish with: kaggle datasets create -p {out} -r zip")


if __name__ == "__main__":
    main()
