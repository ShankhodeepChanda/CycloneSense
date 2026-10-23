"""Tests for the trajectory forecast engine and its HTTP surface."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.trajectory import (
    BAY_OF_BENGAL,
    CoastlineIndex,
    TrajectoryEngine,
    classify_region,
    pressure_from_wind,
    wind_from_pressure,
)

client = TestClient(app)

# The four real AMPHAN fixes from kaggle_trial/tracks.csv, 2020-05-17 18:00 to 2020-05-18 12:00,
# while the storm was intensifying over the open Bay of Bengal.
AMPHAN = [
    {"lat": 12.5, "lon": 86.4, "wind_kt": 120, "pressure_hpa": 932, "timestamp": "2020-05-17T18:00:00Z"},
    {"lat": 13.2, "lon": 86.4, "wind_kt": 130, "pressure_hpa": 919, "timestamp": "2020-05-18T00:00:00Z"},
    {"lat": 13.4, "lon": 86.2, "wind_kt": 140, "pressure_hpa": 911, "timestamp": "2020-05-18T06:00:00Z"},
    {"lat": 14.1, "lon": 86.4, "wind_kt": 145, "pressure_hpa": 901, "timestamp": "2020-05-18T12:00:00Z"},
]

# A storm crossing the northern Bay towards the Sundarbans, 166 km offshore at t0.
INBOUND = [
    {"lat": 17.0 + 0.9 * i, "lon": 87.9 + 0.1 * i, "wind_kt": 110, "pressure_hpa": 945}
    for i in range(4)
]

# A weak system drifting in the far southern Bay, 567 km from the nearest coast at t0.
OFFSHORE = [
    {"lat": 6.0 + 0.2 * i, "lon": 87.0, "wind_kt": 45, "pressure_hpa": 995} for i in range(4)
]


def post(payload: dict):
    return client.post("/predict/trajectory", json=payload)


class TestWindPressureRelation:
    def test_pressure_falls_as_wind_rises(self):
        assert pressure_from_wind(40) > pressure_from_wind(90) > pressure_from_wind(140)

    def test_a_category_five_storm_gets_a_plausible_central_pressure(self):
        assert 880.0 <= pressure_from_wind(140) <= 940.0

    def test_wind_from_pressure_inverts_pressure_from_wind(self):
        for wind in (35.0, 65.0, 100.0, 130.0):
            assert wind_from_pressure(pressure_from_wind(wind)) == pytest.approx(wind, rel=0.02)

    def test_calm_wind_stays_near_the_environmental_pressure(self):
        assert pressure_from_wind(0) >= 1005.0


class TestCoastline:
    def test_a_point_on_the_open_bay_is_hundreds_of_km_from_land(self):
        coast = CoastlineIndex()
        assert coast.distance_to_land_km(15.0, 88.0) > 200.0

    def test_a_point_at_the_sundarbans_is_close_to_land(self):
        coast = CoastlineIndex()
        assert coast.distance_to_land_km(21.6, 88.5) < 60.0

    def test_the_nearest_coast_point_is_returned_with_the_distance(self):
        coast = CoastlineIndex()
        distance, (lat, lon) = coast.nearest(19.0, 86.0)
        assert distance > 0
        assert 5.0 < lat < 30.0 and 70.0 < lon < 100.0

    def test_distance_matches_the_ibtracs_feature_it_stands_in_for(self):
        """The model was trained on IBTrACS dist2land_km; the API must reproduce it."""
        coast = CoastlineIndex()
        # 2020-05-18 12:00 AMPHAN fix: IBTrACS records 520 km to land.
        assert abs(coast.distance_to_land_km(14.1, 86.4) - 520.0) < 80.0


class TestCoastlineCoverage:
    """public/coast.json is an Eastern Hemisphere coastline: it holds no American coast.

    No point on Earth is more than about 2 690 km from land (Point Nemo), so a nearest
    coast beyond that proves the index does not cover that position. Reporting the number
    anyway would put a confidently wrong distance into a forecast.
    """

    def test_a_position_the_coastline_does_not_cover_returns_no_distance(self):
        coast = CoastlineIndex()
        # Off Manzanillo, Mexico:真 nearest land is about 40 km away, but this file has none.
        assert coast.distance_to_land_km(18.77, -104.5) is None
        assert coast.covers(18.77, -104.5) is False

    def test_the_florida_strait_is_also_outside_this_coastline(self):
        assert CoastlineIndex().distance_to_land_km(25.0, -80.0) is None

    def test_the_north_indian_ocean_is_well_inside_coverage(self):
        coast = CoastlineIndex()
        assert coast.covers(14.1, 86.4) is True
        assert coast.distance_to_land_km(14.1, 86.4) is not None

    def test_the_eastern_atlantic_is_covered_because_africa_is_present(self):
        assert CoastlineIndex().distance_to_land_km(28.0, -18.0) is not None


class TestForecastOutsideCoastlineCoverage:
    WESTERN = [{"lat": 15.7 + 0.5 * i, "lon": -104.1 - 0.1 * i, "wind_kt": 105} for i in range(4)]

    def test_an_unavailable_distance_is_declared_not_guessed(self):
        result = TrajectoryEngine().forecast("VANCE", self.WESTERN)
        assert any("dist2land_km" in entry and "unavailable" in entry for entry in result["inputs_imputed"])
        assert result["current_metrics"]["dist2land_km"] is None

    def test_waypoints_report_no_distance_rather_than_a_wrong_one(self):
        result = TrajectoryEngine().forecast("VANCE", self.WESTERN)
        assert all(waypoint["dist2land_km"] is None for waypoint in result["waypoints"])

    def test_landfall_says_it_could_not_be_evaluated(self):
        landfall = TrajectoryEngine().forecast("VANCE", self.WESTERN)["landfall_estimate"]
        assert landfall["projected"] is False
        assert "coastline" in landfall["note"].lower()
        assert "km offshore" not in landfall["note"]

    def test_the_forecast_itself_still_comes_back(self):
        result = TrajectoryEngine().forecast("VANCE", self.WESTERN)
        assert len(result["waypoints"]) == 8
        assert all(math.isfinite(waypoint["lat"]) for waypoint in result["waypoints"])

    def test_the_endpoint_serves_it_without_a_schema_error(self):
        response = post({"cyclone_id": "VANCE", "observations": self.WESTERN})
        assert response.status_code == 200
        assert response.json()["waypoints"][0]["dist2land_km"] is None


class TestRegionClassification:
    def test_the_bay_of_bengal_box_is_recognised(self):
        assert classify_region(15.0, 88.0) == "Bay of Bengal"
        assert BAY_OF_BENGAL.contains(15.0, 88.0)

    def test_the_arabian_sea_is_named_separately(self):
        assert classify_region(17.0, 66.0) == "Arabian Sea"

    def test_a_storm_outside_the_north_indian_ocean_is_not_claimed_as_local(self):
        assert "North Indian Ocean" in classify_region(20.0, 135.0)


class TestEngineWithTrainedWeights:
    def test_the_shipped_checkpoint_is_used_when_present(self):
        engine = TrajectoryEngine()
        if not engine.available:
            pytest.skip("No trajectory checkpoint on this machine; covered by the fallback tests.")
        result = engine.forecast("AMPHAN", AMPHAN)
        assert result["source"] == "trained-sequence-model"
        assert result["model"].startswith("BayOfBengalCycloneForecaster")

    def test_waypoints_cover_every_trained_lead_time(self):
        engine = TrajectoryEngine()
        result = engine.forecast("AMPHAN", AMPHAN)
        assert [w["forecast_hour"] for w in result["waypoints"]] == [6, 12, 18, 24, 30, 36, 42, 48]

    def test_uncertainty_grows_with_lead_time(self):
        engine = TrajectoryEngine()
        radii = [w["uncertainty_radius_km"] for w in engine.forecast("AMPHAN", AMPHAN)["waypoints"]]
        assert radii == sorted(radii)
        assert radii[0] > 0

    def test_a_bay_of_bengal_storm_forecast_stays_in_a_believable_box(self):
        engine = TrajectoryEngine()
        result = engine.forecast("AMPHAN", AMPHAN)
        for waypoint in result["waypoints"]:
            assert 0.0 <= waypoint["lat"] <= 32.0
            assert 70.0 <= waypoint["lon"] <= 100.0

    def test_intensity_is_reported_in_knots_and_hectopascals(self):
        engine = TrajectoryEngine()
        for waypoint in engine.forecast("AMPHAN", AMPHAN)["waypoints"]:
            assert 0.0 <= waypoint["wind_kt"] <= 200.0
            assert 870.0 <= waypoint["pressure_hpa"] <= 1015.0


class TestEngineDegradation:
    def test_a_missing_checkpoint_falls_back_to_a_labelled_physics_path(self, tmp_path):
        engine = TrajectoryEngine(checkpoint_path=tmp_path / "absent.pt")
        result = engine.forecast("AMPHAN", AMPHAN)
        assert engine.available is False
        assert result["source"] == "physics-extrapolation"
        assert "extrapolation" in result["model"].lower()
        assert len(result["waypoints"]) == 8
        assert [w["uncertainty_radius_km"] for w in result["waypoints"]] == sorted(
            w["uncertainty_radius_km"] for w in result["waypoints"]
        )

    def test_a_single_observation_still_produces_a_forecast(self):
        engine = TrajectoryEngine()
        result = engine.forecast("SOLO", [AMPHAN[-1]])
        assert len(result["waypoints"]) == 8
        assert result["observations_used"] == 1
        assert "padded_input_steps" in result["inputs_imputed"]

    def test_missing_pressure_is_derived_and_declared(self):
        engine = TrajectoryEngine()
        observations = [{k: v for k, v in o.items() if k != "pressure_hpa"} for o in AMPHAN]
        result = engine.forecast("AMPHAN", observations)
        assert "pressure_hpa" in result["inputs_imputed"]
        assert result["current_metrics"]["pressure_hpa"] < 1000.0

    def test_missing_wind_is_derived_from_pressure_and_declared(self):
        engine = TrajectoryEngine()
        observations = [{k: v for k, v in o.items() if k != "wind_kt"} for o in AMPHAN]
        result = engine.forecast("AMPHAN", observations)
        assert "wind_kt" in result["inputs_imputed"]
        assert result["current_metrics"]["wind_kt"] > 0

    def test_an_observation_with_neither_wind_nor_pressure_is_refused(self):
        engine = TrajectoryEngine()
        with pytest.raises(ValueError, match="wind|pressure"):
            engine.forecast("NO-INTENSITY", [{"lat": 15.0, "lon": 88.0}])

    def test_no_observations_at_all_is_refused(self):
        engine = TrajectoryEngine()
        with pytest.raises(ValueError, match="observation"):
            engine.forecast("EMPTY", [])

    def test_a_southern_hemisphere_storm_is_forecast_and_flagged_out_of_basin(self):
        engine = TrajectoryEngine()
        southern = [{"lat": -18.0 + 0.4 * i, "lon": 58.0 + 0.3 * i, "wind_kt": 70} for i in range(4)]
        result = engine.forecast("SOUTHERN", southern)
        assert "North Indian Ocean" in result["region"]
        assert all(math.isfinite(w["lat"]) and math.isfinite(w["lon"]) for w in result["waypoints"])


class TestMultiMetricSummary:
    """The dashboard panel reports change over the forecast, so change must be signed."""

    def test_a_weakening_storm_reports_a_rising_pressure_and_falling_wind(self):
        summary = TrajectoryEngine().forecast("AMPHAN", AMPHAN)["summary"]
        assert summary["pressure_change_hpa"] > 0
        assert summary["wind_change_kt"] < 0

    def test_net_change_is_measured_against_the_final_waypoint(self):
        result = TrajectoryEngine().forecast("AMPHAN", AMPHAN)
        final = result["waypoints"][-1]
        current = result["current_metrics"]
        assert result["summary"]["pressure_change_hpa"] == pytest.approx(
            final["pressure_hpa"] - current["pressure_hpa"], abs=0.11
        )
        assert result["summary"]["wind_change_kt"] == pytest.approx(
            final["wind_kt"] - current["wind_kt"], abs=0.11
        )

    def test_deepest_point_on_the_path_is_reported_separately(self):
        summary = TrajectoryEngine().forecast("AMPHAN", AMPHAN)["summary"]
        assert summary["max_deepening_hpa"] <= 0
        assert summary["peak_wind_kt"] >= 0
        assert summary["min_pressure_hpa"] <= summary["peak_pressure_hpa"]

    def test_an_intensifying_storm_reports_a_falling_pressure(self):
        intensifying = [
            {"lat": 10.0 + 0.3 * i, "lon": 87.0 - 0.2 * i, "wind_kt": 35 + 8 * i, "pressure_hpa": 1000 - 7 * i}
            for i in range(4)
        ]
        summary = TrajectoryEngine().forecast("SPINUP", intensifying)["summary"]
        assert summary["max_deepening_hpa"] < 0


class TestLandfall:
    def test_a_storm_aimed_at_the_sundarbans_projects_a_landfall(self):
        engine = TrajectoryEngine()
        landfall = engine.forecast("INBOUND", INBOUND)["landfall_estimate"]
        assert landfall["projected"] is True
        assert 0 < landfall["eta_hours"] <= 48
        assert landfall["location"]
        assert landfall["lat"] is not None and landfall["lon"] is not None

    def test_a_storm_heading_out_to_sea_projects_no_landfall(self):
        engine = TrajectoryEngine()
        landfall = engine.forecast("OFFSHORE", OFFSHORE)["landfall_estimate"]
        assert landfall["projected"] is False
        assert landfall["eta_hours"] is None

    def test_the_landfall_location_names_a_real_coastal_region(self):
        engine = TrajectoryEngine()
        location = engine.forecast("INBOUND", INBOUND)["landfall_estimate"]["location"]
        assert any(token in location for token in ("Bengal", "Bangladesh", "Odisha", "Sundarbans", "coast"))


class TestTrajectoryEndpoint:
    def test_the_response_matches_the_documented_shape(self):
        response = post({"cyclone_id": "AMPHAN", "observations": AMPHAN})
        assert response.status_code == 200
        body = response.json()
        for key in (
            "cyclone_id",
            "region",
            "forecast_generated_at",
            "current_metrics",
            "waypoints",
            "landfall_estimate",
        ):
            assert key in body
        assert body["cyclone_id"] == "AMPHAN"
        assert body["region"] == "Bay of Bengal"
        waypoint = body["waypoints"][0]
        for key in ("forecast_hour", "lat", "lon", "wind_kt", "pressure_hpa", "uncertainty_radius_km"):
            assert key in waypoint

    def test_the_response_declares_where_the_numbers_came_from(self):
        body = post({"cyclone_id": "AMPHAN", "observations": AMPHAN}).json()
        assert body["source"] in {"trained-sequence-model", "physics-extrapolation"}
        assert body["disclaimer"]
        assert isinstance(body["inputs_imputed"], list)

    def test_current_metrics_echo_the_latest_observation(self):
        body = post({"cyclone_id": "AMPHAN", "observations": AMPHAN}).json()
        assert body["current_metrics"]["wind_kt"] == pytest.approx(145.0)
        assert body["current_metrics"]["pressure_hpa"] == pytest.approx(901.0)

    def test_an_empty_observation_list_is_rejected_by_validation(self):
        assert post({"cyclone_id": "AMPHAN", "observations": []}).status_code == 422

    def test_an_out_of_range_latitude_is_rejected_by_validation(self):
        bad = [{"lat": 991.0, "lon": 88.0, "wind_kt": 90}]
        assert post({"cyclone_id": "BAD", "observations": bad}).status_code == 422

    def test_an_observation_without_any_intensity_returns_a_readable_error(self):
        response = post({"cyclone_id": "BAD", "observations": [{"lat": 15.0, "lon": 88.0}]})
        assert response.status_code == 422
        assert "wind" in response.text.lower() or "pressure" in response.text.lower()


class TestExistingContractsStillHold:
    def test_the_legacy_forecast_endpoint_keeps_its_shape(self):
        payload = {
            "cyclone_id": "ARB-2026-02",
            "observations": [
                {"lat": 14.8, "lon": 72.1, "wind_kts": 45, "pressure_hpa": 992},
                {"lat": 15.9, "lon": 71.85, "wind_kts": 55, "pressure_hpa": 984},
                {"lat": 17.15, "lon": 71.5, "wind_kts": 70, "pressure_hpa": 970},
                {"lat": 18.42, "lon": 71.18, "wind_kts": 95, "pressure_hpa": 954},
            ],
        }
        response = client.post("/predict/forecast", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert len(body["prognostic_trajectory"]) == 5
        assert [p["tau_hours"] for p in body["prognostic_trajectory"]] == [12, 24, 36, 48, 72]
        cones = [p["cone_radius_km"] for p in body["prognostic_trajectory"]]
        assert cones == sorted(cones)
        assert body["rapid_intensification_detected"] is True

    def test_the_geojson_export_still_builds_a_line(self):
        body = client.get("/export/geojson?cyclone_id=demo").json()
        assert body["type"] == "FeatureCollection"
        assert len(body["features"][0]["geometry"]["coordinates"]) >= 2

    def test_health_reports_the_trajectory_model_state(self):
        body = client.get("/health").json()
        assert "trajectory_sequence_model" in body["models_loaded"]
