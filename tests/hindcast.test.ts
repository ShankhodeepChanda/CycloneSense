// Run: npm run test:web
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  describeProvenance,
  haversineKm,
  hindcastActual,
  hindcastObservations,
  scoreHindcast,
  type DemoFix,
  type DemoStorm,
  type Waypoint,
} from "../src/trajectory";

const fix = (i: number, lat: number, lon: number, wind = 90, pressure = 960): DemoFix => ({
  time: new Date(Date.UTC(2020, 4, 18, 0, 0) + i * 6 * 3600_000).toISOString(),
  lat,
  lon,
  wind_kt: wind,
  pressure_hpa: pressure,
  dist2land_km: 300,
  category: "VSCS",
});

// A straight northward storm: 0.5 degrees of latitude every 6 hours.
const STORM: DemoStorm = {
  id: "TEST", name: "TESTSTORM", provenance: "held-out", season: 2020, basin: "NI",
  peak_wind_kt: 145, min_pressure_hpa: 901,
  fixes: Array.from({ length: 20 }, (_, i) => fix(i, 10 + 0.5 * i, 86)),
  landfall_index: 16,
  default_start_index: 8,
  usable_start_range: [3, 11],
};

const waypoint = (hour: number, lat: number, lon: number): Waypoint => ({
  forecast_hour: hour, lat, lon, wind_kt: 90, pressure_hpa: 960,
  uncertainty_radius_km: 30, dist2land_km: 300,
});

test("one degree of latitude is about 111 km", () => {
  assert.ok(Math.abs(haversineKm(0, 0, 1, 0) - 111.19) < 1);
  assert.equal(haversineKm(17.5, 86.2, 17.5, 86.2), 0);
});

test("the hindcast input is the four fixes ending at the start point", () => {
  const observations = hindcastObservations(STORM, 8);
  assert.equal(observations.length, 4);
  assert.deepEqual(observations.map((o) => o.lat), [12.5, 13, 13.5, 14]);
  assert.equal(observations[3].wind_kt, 90);
  assert.ok(observations[3].timestamp);
});

test("a start point too early is clamped rather than producing a short window", () => {
  const observations = hindcastObservations(STORM, 1);
  assert.equal(observations.length, 2);
  assert.deepEqual(observations.map((o) => o.lat), [10, 10.5]);
});

test("the actual future is the eight fixes after the start point", () => {
  const actual = hindcastActual(STORM, 8);
  assert.equal(actual.length, 8);
  assert.deepEqual(actual.map((f) => f.lat), [14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18]);
});

test("the actual future truncates at the end of the track", () => {
  assert.equal(hindcastActual(STORM, 15).length, 4);
});

test("a forecast that matches the outcome scores zero error", () => {
  const actual = hindcastActual(STORM, 8);
  const waypoints = actual.map((f, i) => waypoint((i + 1) * 6, f.lat, f.lon));
  const score = scoreHindcast(waypoints, actual);
  assert.deepEqual(score.leads, [6, 12, 18, 24, 30, 36, 42, 48]);
  assert.ok(score.errorKm.every((km) => km !== null && km < 0.001));
  assert.ok(score.meanErrorKm !== null && score.meanErrorKm < 0.001);
});

test("a one degree miss at every lead scores about 111 km", () => {
  const actual = hindcastActual(STORM, 8);
  const waypoints = actual.map((f, i) => waypoint((i + 1) * 6, f.lat + 1, f.lon));
  const score = scoreHindcast(waypoints, actual);
  assert.ok(score.errorKm.every((km) => km !== null && Math.abs(km - 111.19) < 1));
  assert.ok(Math.abs((score.meanErrorKm ?? 0) - 111.19) < 1);
});

test("leads with no observed outcome score null instead of zero", () => {
  const actual = hindcastActual(STORM, 15); // only 4 future fixes remain
  const waypoints = Array.from({ length: 8 }, (_, i) => waypoint((i + 1) * 6, 20, 86));
  const score = scoreHindcast(waypoints, actual);
  assert.equal(score.errorKm.filter((km) => km === null).length, 4);
  assert.equal(score.errorKm.filter((km) => km !== null).length, 4);
});

test("intensity error is reported alongside track error", () => {
  const actual = hindcastActual(STORM, 8);
  const waypoints = actual.map((f, i) => ({ ...waypoint((i + 1) * 6, f.lat, f.lon), wind_kt: 110 }));
  const score = scoreHindcast(waypoints, actual);
  assert.ok(score.windErrorKt.every((kt) => kt !== null && Math.abs(kt - 20) < 0.001));
});

test("the committed demo data is usable by these helpers", () => {
  const data = JSON.parse(readFileSync("public/demo-storms.json", "utf8"));
  assert.ok(data.storms.length >= 3);
  for (const storm of data.storms as DemoStorm[]) {
    const observations = hindcastObservations(storm, storm.default_start_index);
    assert.equal(observations.length, 4, storm.name);
    assert.ok(observations.every((o) => Number.isFinite(o.lat) && Number.isFinite(o.lon)));
    assert.equal(hindcastActual(storm, storm.default_start_index).length, 8, storm.name);
  }
});

test("every demo storm reaches landfall within the forecast window", () => {
  const data = JSON.parse(readFileSync("public/demo-storms.json", "utf8"));
  for (const storm of data.storms as DemoStorm[]) {
    if (storm.landfall_index === null) continue;
    const lead = (storm.landfall_index - storm.default_start_index) * 6;
    assert.ok(lead > 0 && lead <= 48, `${storm.name} landfall at +${lead}h`);
  }
});

test("provenance is described so an in-sample result cannot be mistaken for skill", () => {
  assert.match(describeProvenance("held-out"), /never saw/i);
  assert.match(describeProvenance("trained-on"), /in-sample|flatters/i);
  assert.match(describeProvenance("unknown"), /not recorded/i);
});

test("the committed demo set offers a storm the model never saw", () => {
  const data = JSON.parse(readFileSync("public/demo-storms.json", "utf8"));
  const held = data.storms.filter((s: any) => s.provenance === "held-out");
  assert.ok(held.length >= 1, "demo set must include a held-out storm");
  assert.ok(data.storms.some((s: any) => s.provenance === "trained-on"));
});
