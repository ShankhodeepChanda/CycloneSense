from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_dataset as builder


def main():
    start_year, end_year = 2021, 2023
    storms_per_year = 5
    out = ROOT / "kaggle_temporal_test"
    cache = ROOT / "data" / "raw" / "ibtracs.since1980.list.v04r01.csv"

    if out.exists():
        raise FileExistsError(
            f"{out} already exists. Refusing to overwrite it."
        )
    if not cache.exists():
        raise FileNotFoundError(f"IBTrACS cache not found: {cache}")

    tracks = builder.load_tracks(cache, start_year, end_year)
    if tracks.empty:
        raise RuntimeError("No track records found for 2021-2023.")

    jobs = builder.plan_chips(tracks, 10.0)
    if not jobs:
        raise RuntimeError("No eligible satellite observations found.")

    job_df = pd.DataFrame([
        {
            "sid": j["sid"],
            "year": pd.Timestamp(j["time"]).year,
        }
        for j in jobs
    ])

    counts = (
        job_df.groupby(["year", "sid"])
        .size()
        .rename("planned_jobs")
        .reset_index()
    )

    selected = set()
    print("\nDeterministic storm selection:")
    for year in range(start_year, end_year + 1):
        candidates = counts[counts["year"] == year].sort_values(
            ["planned_jobs", "sid"],
            ascending=[False, True],
        )
        chosen = candidates.head(storms_per_year)

        print(f"\n{year}: selected {len(chosen)} storms")
        for row in chosen.itertuples(index=False):
            print(f"  {row.sid}: {row.planned_jobs} planned observations")
            selected.add(row.sid)

    if not selected:
        raise RuntimeError("No storms selected.")

    selected_tracks = tracks[tracks["sid"].isin(selected)].copy()
    selected_tracks["split"] = "temporal_test"

    selected_jobs = []
    for job in jobs:
        if job["sid"] in selected:
            copy = dict(job)
            copy["split"] = "temporal_test"
            selected_jobs.append(copy)

    print(f"\nSelected storms: {len(selected)}")
    print(f"Planned IR image requests: {len(selected_jobs)}")
    print(f"Output: {out}")

    out.mkdir(parents=True)
    selected_tracks.to_csv(out / "tracks.csv", index=False)

    session = builder.make_session()
    rows = []
    failed = 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                builder.do_job,
                session,
                job,
                "ir",
                out,
                10.0,
                256,
                0.6,
            )
            for job in selected_jobs
        ]

        for number, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as exc:
                print(f"Request error: {exc}")
                row = None

            if row is None:
                failed += 1
            else:
                rows.append(row)

            if number % 20 == 0 or number == len(futures):
                print(
                    f"Completed {number}/{len(futures)} requests; "
                    f"kept {len(rows)}; skipped/failed {failed}"
                )

    index = pd.DataFrame(rows)
    index.to_csv(out / "images_index.csv", index=False)

    print("\nDownload summary")
    print(f"Track rows: {len(selected_tracks)}")
    print(f"Images kept: {len(index)}")
    print(f"Skipped/failed requests: {failed}")

    if not index.empty:
        index["year"] = pd.to_datetime(index["time"]).dt.year
        print("\nImages by year:")
        print(index.groupby("year").size().to_string())
        print("\nUnique storms by year:")
        print(index.groupby("year")["sid"].nunique().to_string())
        print("\nAll images are infrared; no training was performed.")

    print(f"\nDataset saved to: {out}")


if __name__ == "__main__":
    main()
