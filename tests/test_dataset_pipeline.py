"""Offline tests for data/build_dataset.py. The CSV below is a tiny hand-made fixture
in IBTrACS layout (header, units row, data) used only to test the code, never as data."""

import io

import numpy as np
import pandas as pd
from PIL import Image

from data.build_dataset import category, decode_chip, load_tracks, plan_chips, split_of

HEADER = "SID,SEASON,BASIN,NAME,ISO_TIME,TRACK_TYPE,LAT,LON,USA_WIND,USA_PRES,WMO_WIND,WMO_PRES,DIST2LAND"
UNITS = " , Year, , , ,  ,degrees_north,degrees_east,kts,mb,kts,mb,km"


def rows(sid, start, winds, lat0=15.0, lon0=88.0):
    out = []
    for i, w in enumerate(winds):
        t = pd.Timestamp(start) + pd.Timedelta(hours=6 * i)
        out.append(f"{sid},2024,NI,TEST,{t:%Y-%m-%d %H:%M:%S},main,{lat0 + 0.3 * i},{lon0 - 0.2 * i},{w},990, ,  ,200")
    return out


def write_fixture(tmp_path, extra=()):
    lines = [HEADER, UNITS]
    lines += rows("2024001N15088", "2024-05-24 00:00", [30, 35, 45, 60, 80, 95, 110])
    lines += extra
    # a 3-hourly interpolated row and a non-main row that must be ignored
    lines.append("2024001N15088,2024,NI,TEST,2024-05-24 03:00:00,main,15.1,87.9,32,991, ,  ,200")
    lines.append("2024001N15088,2024,NI,TEST,2024-05-24 06:00:00,spur-other,15.1,87.9,32,991, ,  ,200")
    path = tmp_path / "ib.csv"
    path.write_text("\n".join(lines))
    return path


def test_category_boundaries():
    assert [category(k) for k in (20, 34, 64, 83, 96, 113, 137, 170)] == ["TD", "TS", "C1", "C2", "C3", "C4", "C5", "C5"]


def test_split_is_deterministic_and_roughly_70_15_15():
    assert split_of("abc") == split_of("abc")
    counts = pd.Series([split_of(f"storm{i}") for i in range(5000)]).value_counts(normalize=True)
    assert abs(counts["train"] - 0.70) < 0.03 and abs(counts["val"] - 0.15) < 0.03


def test_load_tracks_filters_and_labels(tmp_path):
    slow = rows("2024002N12085", "2024-06-01 00:00", [30, 32, 34, 36, 38, 40])
    df = load_tracks(write_fixture(tmp_path, slow), 2020, 2025)
    fast = df[df["sid"] == "2024001N15088"].reset_index(drop=True)
    assert len(fast) == 7  # 3-hourly and non-main rows dropped
    assert np.isnan(fast.loc[0, "dwind"]) and fast.loc[1, "dwind"] == 5
    assert list(fast["ri_24h"].iloc[:3]) == [1.0, 1.0, 1.0]  # +50, +60, +65 kt within 24 h
    assert fast["ri_24h"].iloc[3:].isna().all()  # no +24 h record
    assert df[df["sid"] == "2024002N12085"]["ri_24h"].iloc[0] == 0.0  # +8 kt is not RI


def test_year_filter_and_missing_wind_dropped(tmp_path):
    extra = ["2019001N10080,2019,NI,OLD,2019-05-01 00:00:00,main,10,80, ,990, ,  ,100"]
    df = load_tracks(write_fixture(tmp_path, extra), 2020, 2025)
    assert "2019001N10080" not in set(df["sid"])


def test_plan_chips_one_per_storm_day_near_overpass(tmp_path):
    tracks = load_tracks(write_fixture(tmp_path), 2020, 2025)
    jobs = plan_chips(tracks, 10.0)
    keys = {(j["sat"], j["sid"], j["date"]) for j in jobs}
    assert len(keys) == len(jobs)  # no duplicate storm-day per satellite
    for j in jobs:
        assert abs(j["offset_h"]) <= 3.0
    # Terra local 10:30 at ~88E is ~04:40 UTC, so the 06 UTC record is the nearest synoptic time
    terra = [j for j in jobs if j["sat"] == "terra" and j["date"] == "2024-05-24"]
    assert terra and terra[0]["time"].hour == 6


def test_decode_chip_valid_fraction():
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    for x in range(5):
        for y in range(10):
            img.putpixel((x, y), (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _, valid = decode_chip(buf.getvalue())
    assert valid == 0.5

    rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 0))  # fully transparent = no data
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    assert decode_chip(buf.getvalue())[1] == 0.0


def test_do_job_saves_chip_and_resumes(tmp_path, monkeypatch):
    import data.build_dataset as bd

    calls = []

    def fake_fetch(session, layer, date, lat, lon, chip_deg, size):
        calls.append(layer)
        return Image.new("RGB", (size, size), (120, 120, 120)), 0.9

    monkeypatch.setattr(bd, "fetch_chip", fake_fetch)
    tracks = load_tracks(write_fixture(tmp_path), 2020, 2025)
    job = plan_chips(tracks, 10.0)[0]

    row = bd.do_job(None, job, "ir", tmp_path, 10.0, 64, 0.6)
    assert (tmp_path / row["image_path"]).exists()
    assert row["channel"] == "ir" and row["split"] == job["split"] and row["wind_kt"] == job["wind_kt"]
    assert calls == [bd.SATELLITES[job["sat"]][3]]  # IR layer used for the ir channel

    bd.do_job(None, job, "ir", tmp_path, 10.0, 64, 0.6)
    assert len(calls) == 1  # second run reuses the file, no new download

    monkeypatch.setattr(bd, "fetch_chip", lambda *a: (Image.new("RGB", (64, 64)), 0.1))
    assert bd.do_job(None, job, "truecolor", tmp_path, 10.0, 64, 0.6) is None  # too little data -> dropped


def test_kaggle_metadata_limits(tmp_path):
    from data.build_dataset import write_metadata

    write_metadata(tmp_path, "someone", pd.DataFrame(), pd.DataFrame())
    meta = __import__("json").loads((tmp_path / "dataset-metadata.json").read_text())
    assert 6 <= len(meta["title"]) <= 50 and 20 <= len(meta["subtitle"]) <= 80
    assert meta["id"] == "someone/cyclonesense-tc-satellite"
