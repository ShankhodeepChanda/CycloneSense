"""Tests for the hindcast demo dataset.

The dashboard's demo mode replays a real storm: it forecasts from a chosen fix and draws
the track the storm actually took, so the error is visible rather than asserted. That
only works if the exported fixes are genuine best-track data at the cadence the model
expects, with enough history before the start point and enough future after it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_demo_storms import (
    DEMO_STORM_NAMES,
    FUTURE_FIXES,
    INPUT_FIXES,
    build_demo_storms,
    export_demo_storms,
)

TRACKS = "kaggle_trial/tracks.csv"


@pytest.fixture(scope="module")
def payload():
    return build_demo_storms(TRACKS)


class TestStormSelection:
    def test_every_requested_storm_is_present(self, payload):
        assert {storm["name"] for storm in payload["storms"]} == set(DEMO_STORM_NAMES)

    def test_all_are_north_indian_ocean_storms(self, payload):
        assert {storm["basin"] for storm in payload["storms"]} == {"NI"}

    def test_the_source_is_named_so_the_numbers_are_traceable(self, payload):
        assert "IBTrACS" in payload["source"]
        assert TRACKS in payload["source"]

    def test_each_storm_carries_its_peak_intensity(self, payload):
        for storm in payload["storms"]:
            assert 40 <= storm["peak_wind_kt"] <= 200
            assert 880 <= storm["min_pressure_hpa"] <= 1010


class TestFixes:
    def test_fixes_are_six_hourly_and_contiguous(self, payload):
        from datetime import datetime

        for storm in payload["storms"]:
            times = [datetime.fromisoformat(fix["time"]) for fix in storm["fixes"]]
            gaps = {(b - a).total_seconds() / 3600 for a, b in zip(times, times[1:])}
            assert gaps == {6.0}, f"{storm['name']} has gaps {gaps}"

    def test_every_fix_has_a_position_and_an_intensity(self, payload):
        for storm in payload["storms"]:
            for fix in storm["fixes"]:
                assert -90 <= fix["lat"] <= 90
                assert -180 <= fix["lon"] <= 180
                assert fix["wind_kt"] is not None
                assert fix["pressure_hpa"] is not None

    def test_positions_are_in_the_bay_of_bengal_region(self, payload):
        """These are the demo cases, so they should look like Bay of Bengal storms."""
        for storm in payload["storms"]:
            inside = sum(
                1 for fix in storm["fixes"] if 5 <= fix["lat"] <= 25 and 78 <= fix["lon"] <= 95
            )
            assert inside / len(storm["fixes"]) > 0.6, storm["name"]


class TestStartPoint:
    def test_the_default_start_leaves_enough_history_and_future(self, payload):
        for storm in payload["storms"]:
            start = storm["default_start_index"]
            assert start >= INPUT_FIXES - 1, storm["name"]
            assert start + FUTURE_FIXES < len(storm["fixes"]), storm["name"]

    def test_every_offered_start_is_usable(self, payload):
        """The UI lets the user move the start point; none of them may be invalid."""
        for storm in payload["storms"]:
            assert storm["usable_start_range"][0] == INPUT_FIXES - 1
            assert storm["usable_start_range"][1] == len(storm["fixes"]) - FUTURE_FIXES - 1
            assert storm["usable_start_range"][0] <= storm["default_start_index"]
            assert storm["default_start_index"] <= storm["usable_start_range"][1]

    def test_the_default_start_is_set_two_days_before_landfall(self, payload):
        """The interesting forecast is the one that has to call the landfall."""
        for storm in payload["storms"]:
            if storm["landfall_index"] is None:
                continue
            lead = storm["landfall_index"] - storm["default_start_index"]
            assert 0 < lead <= FUTURE_FIXES, f"{storm['name']} lead {lead}"

    def test_landfall_index_points_at_a_fix_that_reached_the_coast(self, payload):
        for storm in payload["storms"]:
            if storm["landfall_index"] is None:
                continue
            assert storm["fixes"][storm["landfall_index"]]["dist2land_km"] == 0


class TestExport:
    def test_export_writes_json_the_browser_can_fetch(self, tmp_path):
        target = export_demo_storms(TRACKS, tmp_path / "demo-storms.json")
        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert len(loaded["storms"]) == len(DEMO_STORM_NAMES)

    def test_the_file_stays_small_enough_to_ship_to_the_browser(self, tmp_path):
        target = export_demo_storms(TRACKS, tmp_path / "demo-storms.json")
        assert target.stat().st_size < 120_000, "demo data should stay well under 120 KB"

    def test_the_committed_copy_matches_what_the_script_produces(self):
        """A stale committed copy would demo different numbers than the data supports."""
        committed = Path("public/demo-storms.json")
        if not committed.is_file():
            pytest.skip("Run python -m scripts.export_demo_storms first.")
        assert json.loads(committed.read_text(encoding="utf-8")) == build_demo_storms(TRACKS)


class TestProvenance:
    """A forecast of a storm the model fitted is not evidence of skill, so the demo has
    to say which is which and must offer at least one case the model never saw."""

    def test_every_storm_declares_whether_the_model_trained_on_it(self, payload):
        for storm in payload["storms"]:
            assert storm["provenance"] in {"trained-on", "held-out", "unknown"}

    def test_at_least_one_held_out_storm_is_offered(self, payload):
        held = [s["name"] for s in payload["storms"] if s["provenance"] == "held-out"]
        assert held, "the demo must include a storm the model never saw"

    def test_the_labels_match_the_shipped_checkpoint(self, payload):
        from scripts.export_demo_storms import _training_split

        trained_on, held_out = _training_split()
        if not trained_on and not held_out:
            pytest.skip("No exported checkpoint on this machine.")
        for storm in payload["storms"]:
            if storm["provenance"] == "trained-on":
                assert storm["id"] in trained_on, storm["name"]
            elif storm["provenance"] == "held-out":
                assert storm["id"] in held_out, storm["name"]

    def test_the_distinction_is_explained_in_the_payload(self, payload):
        assert "in-sample" in payload["provenance_note"]

    def test_a_set_with_no_held_out_storm_is_refused(self, monkeypatch):
        """The guard must fail the export, not quietly ship a flattering demo."""
        import scripts.export_demo_storms as module

        monkeypatch.setattr(module, "DEMO_STORM_NAMES", ("AMPHAN", "FANI"))
        with pytest.raises(ValueError, match="held-out"):
            module.build_demo_storms(TRACKS)
