"""Export real Bay of Bengal storms for the dashboard's hindcast demo.

The dashboard can only show a forecast when GDACS reports a live storm, and for most of
the year it reports none. That hides the whole feature. Replaying a past storm fixes
that, and it is a better demonstration anyway: a historical case has a known outcome, so
the map can draw the forecast *and* the track the storm actually took, and the audience
reads the error off the screen instead of taking a metric on trust.

The fixes are genuine IBTrACS best-track observations from the repository's own data, cut
to runs that are exactly 6-hourly because that is the cadence the model's motion features
assume. The default start point sits one forecast horizon before landfall, so the case on
screen is the one that has to call where and when the storm comes ashore.

Usage:
    python -m scripts.export_demo_storms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: Bay of Bengal landfalling storms spanning the intensity range. The last two were held
#: out of training, which matters for the demo: a forecast of a storm the model fitted is
#: not evidence of skill, so the set has to include cases it never saw and each storm is
#: labelled with which it is.
DEMO_STORM_NAMES: tuple[str, ...] = ("AMPHAN", "FANI", "TITLI", "NIVAR", "GAJA", "BUREVI")

#: Exported weights carry the training and validation storm ids.
CHECKPOINT = Path("models/bob_trajectory_gru.npz")

#: Fixes the model consumes as input, and forecast steps it emits.
INPUT_FIXES = 4
FUTURE_FIXES = 8

STEP_HOURS = 6
DEFAULT_OUTPUT = Path("public/demo-storms.json")


def _longest_six_hourly_run(group) -> Any:
    """The longest stretch of fixes spaced exactly 6 hours apart.

    IBTrACS mixes cadences around landfall, and a window built across a gap would hand
    the model motion it never observed.
    """
    import pandas as pd

    group = group.sort_values("time").reset_index(drop=True)
    gaps = group["time"].diff()
    breaks = (gaps != pd.Timedelta(hours=STEP_HOURS)) & gaps.notna()
    runs = [run for _, run in group.groupby(breaks.cumsum(), sort=True)]
    return max(runs, key=len).reset_index(drop=True)


def _training_split(checkpoint: str | Path = CHECKPOINT) -> tuple[set[str], set[str]]:
    """Storm ids the shipped model trained on, and those held out of it."""
    path = Path(checkpoint)
    if not path.is_file():
        return set(), set()
    from ml.numpy_forecaster import NumpyForecaster

    _, meta = NumpyForecaster.load(path)
    return set(meta.get("train_storms") or []), set(meta.get("val_storms") or [])


def build_demo_storms(tracks: str | Path = "kaggle_trial/tracks.csv") -> dict[str, Any]:
    """Build the demo payload from best-track data."""
    import pandas as pd

    trained_on, held_out = _training_split()
    frame = pd.read_csv(tracks)
    frame["time"] = pd.to_datetime(frame["time"])

    storms: list[dict[str, Any]] = []
    for name in DEMO_STORM_NAMES:
        matching = frame[(frame["name"] == name) & (frame["basin"] == "NI")]
        if matching.empty:
            raise ValueError(f"{name} is not in {tracks} as a North Indian Ocean storm.")
        run = _longest_six_hourly_run(matching)
        if len(run) < INPUT_FIXES + FUTURE_FIXES + 1:
            raise ValueError(
                f"{name} has only {len(run)} contiguous 6-hourly fixes; "
                f"a hindcast needs {INPUT_FIXES + FUTURE_FIXES + 1}."
            )

        fixes = [
            {
                "time": row.time.isoformat(),
                "lat": round(float(row.lat), 2),
                "lon": round(float(row.lon), 2),
                "wind_kt": None if pd.isna(row.wind_kt) else round(float(row.wind_kt), 1),
                "pressure_hpa": None if pd.isna(row.pres_hpa) else round(float(row.pres_hpa), 1),
                "dist2land_km": None if pd.isna(row.dist2land_km) else round(float(row.dist2land_km), 1),
                "category": None if pd.isna(row.category) else str(row.category),
            }
            for row in run.itertuples()
        ]

        onshore = [index for index, fix in enumerate(fixes) if fix["dist2land_km"] == 0]
        landfall_index = onshore[0] if onshore else None

        earliest = INPUT_FIXES - 1
        latest = len(fixes) - FUTURE_FIXES - 1
        if landfall_index is not None:
            # One horizon before landfall: the forecast then has to call the crossing.
            preferred = landfall_index - FUTURE_FIXES
        else:
            peak = max(range(len(fixes)), key=lambda i: fixes[i]["wind_kt"] or 0)
            preferred = peak - FUTURE_FIXES
        default_start = max(earliest, min(latest, preferred))

        sid = str(run.loc[0, "sid"])
        if sid in trained_on:
            provenance = "trained-on"
        elif sid in held_out:
            provenance = "held-out"
        else:
            provenance = "unknown"

        storms.append(
            {
                "id": sid,
                "name": name,
                "provenance": provenance,
                "season": int(run.loc[0, "season"]),
                "basin": str(run.loc[0, "basin"]),
                "peak_wind_kt": round(float(run["wind_kt"].max()), 1),
                "min_pressure_hpa": round(float(run["pres_hpa"].min()), 1),
                "fixes": fixes,
                "landfall_index": landfall_index,
                "default_start_index": default_start,
                "usable_start_range": [earliest, latest],
            }
        )

    if not any(storm["provenance"] == "held-out" for storm in storms):
        raise ValueError(
            "The demo set contains no held-out storm. A forecast of a storm the model was "
            "trained on is not evidence of skill, so at least one case it never saw must be "
            "offered alongside the rest."
        )

    return {
        "source": f"NOAA IBTrACS best-track observations via {tracks}",
        "note": (
            "Historical storms, not live data. The dashboard forecasts from a chosen fix "
            "and draws the track the storm actually took, so the forecast error is visible."
        ),
        "provenance_note": (
            "trained-on: the model fitted this storm, so its error is in-sample and flatters "
            "the model. held-out: the model never saw it, so that error is the honest one."
        ),
        "input_fixes": INPUT_FIXES,
        "future_fixes": FUTURE_FIXES,
        "step_hours": STEP_HOURS,
        "storms": storms,
    }


def export_demo_storms(
    tracks: str | Path = "kaggle_trial/tracks.csv",
    out_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Write the demo payload where the frontend can fetch it."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_demo_storms(tracks), indent=1) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tracks", default="kaggle_trial/tracks.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    written = export_demo_storms(args.tracks, args.out)
    payload = json.loads(written.read_text(encoding="utf-8"))
    print(f"wrote {written} ({written.stat().st_size / 1024:.0f} KB)")
    for storm in payload["storms"]:
        start = storm["fixes"][storm["default_start_index"]]
        landfall = (
            storm["fixes"][storm["landfall_index"]]["time"][:16].replace("T", " ")
            if storm["landfall_index"] is not None
            else "none recorded"
        )
        print(
            f"  {storm['name']:<8} {storm['season']}  {len(storm['fixes']):>2} fixes  "
            f"peak {storm['peak_wind_kt']:>5.0f} kt  {storm['provenance']:<10} "
            f"start {start['time'][:16].replace('T', ' ')}  landfall {landfall}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
