from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_dataset as builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--out", default="kaggle_multiyear")
    parser.add_argument("--per-split-year", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache = ROOT / "data" / "raw" / "ibtracs.since1980.list.v04r01.csv"
    out = ROOT / args.out

    if not cache.exists():
        raise FileNotFoundError(f"IBTrACS CSV not found: {cache}")

    tracks = builder.load_tracks(cache, args.start_year, args.end_year)
    if tracks.empty:
        raise RuntimeError("No track records found for the selected years.")

    jobs = builder.plan_chips(tracks, 10.0)
    if not jobs:
        raise RuntimeError("No satellite-image jobs were planned.")

    job_df = pd.DataFrame([
        {"sid": j["sid"], "year": pd.Timestamp(j["time"]).year,
         "split": j["split"]}
        for j in jobs
    ])
    counts = (
        job_df.groupby(["year", "split", "sid"])
        .size().rename("jobs").reset_index()
    )

    selected = set()
    print("\nStorm selection (year / split / SID / planned jobs):")
    for year in range(args.start_year, args.end_year + 1):
        for split in ("train", "val", "test"):
            candidates = counts[
                (counts["year"] == year) & (counts["split"] == split)
            ].sort_values(["jobs", "sid"], ascending=[False, True])
            chosen = candidates.head(args.per_split_year)
            if chosen.empty:
                print(f"{year} / {split}: no eligible storms")
            for row in chosen.itertuples(index=False):
                selected.add(row.sid)
                print(f"{year} / {split} / {row.sid} / {row.jobs}")

    if not selected:
        raise RuntimeError("No storms were selected.")

    selected_tracks = tracks[tracks["sid"].isin(selected)].copy()
    selected_jobs = [j for j in jobs if j["sid"] in selected]

    print("\nSelected unique storms by split:")
    print(
        selected_tracks[["sid", "split"]].drop_duplicates()
        .groupby("split")["sid"].nunique().to_string()
    )
    print(f"\nUnique storms: {selected_tracks['sid'].nunique()}")
    print(f"Storm-day/satellite jobs: {len(selected_jobs)}")
    print(f"Maximum image requests: {len(selected_jobs) * len(builder.CHANNELS)}")
    print(f"Output folder: {out}")

    if args.dry_run:
        print("\nDry run only; no images downloaded.")
        return

    out.mkdir(parents=True, exist_ok=True)
    selected_tracks.to_csv(out / "tracks.csv", index=False)
    session = builder.make_session()
    rows = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(builder.do_job, session, job, channel, out,
                        10.0, 256, 0.6)
            for job in selected_jobs
            for channel in builder.CHANNELS
        ]
        for number, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as exc:
                print(f"Job error: {exc}")
                row = None
            if row is None:
                failed += 1
            else:
                rows.append(row)
            if number % 20 == 0:
                print(f"Completed {number} requests; "
                      f"{len(rows)} images kept; {failed} skipped")

    index = pd.DataFrame(rows)
    index.to_csv(out / "images_index.csv", index=False)
    print("\nDownload finished.")
    print(f"Tracks: {len(selected_tracks)} rows")
    print(f"Images kept: {len(index)}")
    print(f"Requests skipped/failed: {failed}")

    if not index.empty:
        index["year"] = pd.to_datetime(index["time"]).dt.year
        print("\nImages by year and channel:")
        print(index.groupby(["year", "channel"]).size().to_string())
        print("\nImages by split:")
        print(index.groupby("split").size().to_string())
        print("\nUnique storms per split:")
        print(index.groupby("split")["sid"].nunique().to_string())

    print(f"\nDataset saved to: {out}")


if __name__ == "__main__":
    main()
