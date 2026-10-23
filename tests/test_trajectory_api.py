"""Tests for the trajectory forecast engine and its HTTP surface."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.trajectory import (
    BAY_OF_BENGAL,
    CoastlineIndex,
    TrajectoryEngine,
    classify_region,
    nearest_coastal_place,
    prepare_sequence,
    pressure_from_wind,
    wind_from_pressure,
)
from ml.sequence_dataset import FEATURES

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
        # Off Manzanillo, Mexico: the real nearest land is about 40 km away, but this file has none.
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
        # "North Indian Ocean" is a substring of the correct answer AND of the wrong one,
        # so this has to assert the prefix to be able to fail at all.
        assert classify_region(20.0, 135.0) == "Outside the North Indian Ocean"
        assert classify_region(-18.0, 58.0) == "Outside the North Indian Ocean"


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
        assert result["region"] == "Outside the North Indian Ocean"
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


# ===================================================================================
# Fix pass for the final review findings.
# ===================================================================================


class TestFallbackHonesty:
    """With no checkpoint the response must not claim a trained model anywhere."""

    def _fallback(self, tmp_path, observations=None):
        engine = TrajectoryEngine(checkpoint_path=tmp_path / "absent.pt")
        return engine.forecast("AMPHAN", observations or AMPHAN)

    def test_the_disclaimer_does_not_claim_a_trained_model(self, tmp_path):
        disclaimer = self._fallback(tmp_path)["disclaimer"].lower()
        # It may mention that no trained checkpoint is installed; what it must not do is
        # assert that a model was trained on anything and produced this.
        assert "trained on" not in disclaimer
        assert "not a model forecast" in disclaimer

    def test_the_trained_disclaimer_still_says_where_the_numbers_came_from(self):
        engine = TrajectoryEngine()
        if not engine.available:
            pytest.skip("No trajectory checkpoint on this machine.")
        assert "trained" in engine.forecast("AMPHAN", AMPHAN)["disclaimer"].lower()

    def test_holding_wind_constant_reports_no_pressure_change(self, tmp_path):
        """A persistence forecast predicts no intensity change; it must not invent one."""
        steady = [dict(fix, wind_kt=100.0, pressure_hpa=950.0) for fix in AMPHAN]
        result = self._fallback(tmp_path, steady)
        assert result["summary"]["wind_change_kt"] == pytest.approx(0.0, abs=0.11)
        assert result["summary"]["pressure_change_hpa"] == pytest.approx(0.0, abs=0.11)
        assert result["summary"]["max_deepening_hpa"] == pytest.approx(0.0, abs=0.11)

    def test_the_observed_pressure_is_carried_not_recomputed_from_wind(self, tmp_path):
        """901 hPa observed must stay 901 hPa while wind is held, not become 891.6."""
        result = self._fallback(tmp_path)
        assert result["waypoints"][0]["pressure_hpa"] == pytest.approx(901.0, abs=0.5)

    def test_a_weakening_fallback_still_fills_the_pressure(self, tmp_path):
        """Over land the wind decays, so the pressure must rise with it."""
        inbound = [
            {"lat": 19.0 + 0.9 * i, "lon": 88.0, "wind_kt": 110.0, "pressure_hpa": 945.0}
            for i in range(4)
        ]
        result = self._fallback(tmp_path, inbound)
        assert result["summary"]["wind_change_kt"] < 0
        assert result["summary"]["pressure_change_hpa"] > 0


class TestObservationSpacing:
    """The model's motion features are per 6 hours. Supplied timestamps say what the
    real spacing was, and ignoring them silently rescales every forecast."""

    def _at_spacing(self, hours):
        start = 1_589_000_000
        return [
            {
                "lat": 12.5 + 0.6 * i,
                "lon": 86.4 + 0.1 * i,
                "wind_kt": 120.0,
                "pressure_hpa": 930.0,
                "timestamp": datetime.fromtimestamp(start + i * hours * 3600, timezone.utc).isoformat(),
            }
            for i in range(4)
        ]

    def test_twelve_hourly_input_is_rescaled_to_the_six_hourly_step(self):
        engine = TrajectoryEngine()
        six = engine.forecast("A", self._at_spacing(6))
        twelve = engine.forecast("B", self._at_spacing(12))
        # Same positions, half the cadence: the per-6h motion is half, so the +48 h
        # forecast must not run twice as far.
        assert twelve["waypoints"][-1]["lat"] < six["waypoints"][-1]["lat"]

    def test_a_non_six_hourly_sequence_says_it_was_rescaled(self):
        result = TrajectoryEngine().forecast("B", self._at_spacing(12))
        assert any("rescaled" in entry for entry in result["inputs_imputed"])

    def test_a_six_hourly_sequence_is_not_flagged(self):
        result = TrajectoryEngine().forecast("A", self._at_spacing(6))
        assert not any("rescaled" in entry for entry in result["inputs_imputed"])

    def test_without_timestamps_the_assumption_is_declared(self):
        plain = [{"lat": 12.5 + 0.6 * i, "lon": 86.4, "wind_kt": 120.0} for i in range(4)]
        result = TrajectoryEngine().forecast("C", plain)
        assert any("6-hourly" in entry or "six-hourly" in entry for entry in result["inputs_imputed"])

    def test_a_single_observation_needs_no_spacing_claim(self):
        result = TrajectoryEngine().forecast("D", [AMPHAN[-1]])
        assert not any("rescaled" in entry for entry in result["inputs_imputed"])


class TestAntimeridian:
    """A storm crossing the dateline must not hand the model a 359 degree step."""

    CROSSING = [
        {"lat": 14.0, "lon": 179.2, "wind_kt": 80.0},
        {"lat": 14.4, "lon": 179.6, "wind_kt": 80.0},
        {"lat": 14.8, "lon": -179.9, "wind_kt": 80.0},
        {"lat": 15.2, "lon": -179.4, "wind_kt": 80.0},
    ]

    def test_longitude_steps_wrap_instead_of_spanning_the_globe(self):
        prepared = prepare_sequence(self.CROSSING, CoastlineIndex.shared())
        steps = prepared.features[:, FEATURES.index("dlon")]
        assert abs(steps).max() < 2.0, f"unwrapped longitude step: {steps}"

    def test_the_forecast_stays_in_a_believable_longitude_band(self):
        result = TrajectoryEngine().forecast("DATELINE", self.CROSSING)
        for waypoint in result["waypoints"]:
            assert -180.0 <= waypoint["lon"] <= 180.0


class TestLegacyForecastEdges:
    def test_an_empty_observation_list_is_refused_not_invented(self):
        """It used to answer with a hardcoded Arabian Sea storm that does not exist."""
        with pytest.raises(ValueError, match="observation"):
            TrajectoryEngine().legacy_forecast("EMPTY", [])

    def test_rapid_intensification_ignores_observations_with_no_wind(self):
        observations = [
            {"lat": 12.0, "lon": 86.0, "pressure_hpa": 1000.0},  # no wind at all
            {"lat": 13.0, "lon": 86.0, "wind_kt": 60.0},
        ]
        result = TrajectoryEngine().legacy_forecast("NO-RI", observations)
        assert result["rapid_intensification_detected"] is False

    def test_real_intensification_is_still_detected(self):
        observations = [
            {"lat": 12.0, "lon": 86.0, "wind_kt": 45.0},
            {"lat": 13.0, "lon": 86.0, "wind_kt": 95.0},
        ]
        assert TrajectoryEngine().legacy_forecast("RI", observations)["rapid_intensification_detected"] is True

    def test_the_point_beyond_the_trained_horizon_is_marked_as_such(self):
        result = TrajectoryEngine().legacy_forecast("ARB", [dict(fix, wind_kts=fix["wind_kt"]) for fix in AMPHAN])
        by_tau = {point["tau_hours"]: point for point in result["prognostic_trajectory"]}
        assert by_tau[72]["extrapolated_beyond_training"] is True
        assert by_tau[48]["extrapolated_beyond_training"] is False


class TestGazetteerPrecision:
    def test_a_place_name_is_not_used_hundreds_of_km_from_the_crossing(self):
        """Great Nicobar is 398 km from Port Blair; naming it Port Blair is wrong."""
        assert "Port Blair" not in nearest_coastal_place(8.11, 93.46)

    def test_a_coordinate_description_is_used_when_nothing_is_near(self):
        described = nearest_coastal_place(8.11, 93.46)
        assert "8.1" in described and "93.5" in described

    def test_a_genuinely_near_place_is_still_named(self):
        assert "Sundarbans" in nearest_coastal_place(21.65, 88.32) or "Bengal" in nearest_coastal_place(21.65, 88.32)


class TestCoastlineLongitudeCoverage:
    """coast.json holds no vertex between 100W and 20W, so a position whose true
    nearest land is American can still find a European coast inside 2 800 km."""

    def test_the_azores_are_reported_as_uncovered_not_1900_km_offshore(self):
        # About 150 km from land in reality; the nearest vertex in this file is Portugal.
        assert CoastlineIndex().distance_to_land_km(38.5, -28.0) is None

    def test_the_north_indian_ocean_is_still_covered(self):
        assert CoastlineIndex().distance_to_land_km(14.1, 86.4) is not None

    def test_the_eastern_atlantic_near_africa_is_still_covered(self):
        assert CoastlineIndex().distance_to_land_km(28.0, -16.0) is not None


class TestImputationListIsNotDiluted:
    def test_a_correctly_derived_distance_is_not_listed_as_imputed(self):
        result = TrajectoryEngine().forecast("AMPHAN", AMPHAN)
        assert not any(entry.startswith("dist2land_km (derived") for entry in result["inputs_imputed"])

    def test_an_unavailable_distance_is_still_listed(self):
        western = [{"lat": 15.7 + 0.5 * i, "lon": -104.1, "wind_kt": 105.0} for i in range(4)]
        result = TrajectoryEngine().forecast("VANCE", western)
        assert any("dist2land_km" in entry for entry in result["inputs_imputed"])
