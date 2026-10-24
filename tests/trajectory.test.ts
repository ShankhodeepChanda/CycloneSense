// Run: npm run test:web
import test from "node:test";
import assert from "node:assert/strict";
import {
  WAYPOINT_HIGHLIGHTS,
  conePolygon,
  describeLandfall,
  describePressureChange,
  formatPosition,
  isOutsideHomeBasin,
  kmToDegrees,
  observationsFromTrack,
  trackLatLngs,
  type Waypoint,
} from "../src/trajectory";

const waypoint = (hour: number, lat: number, lon: number, radius: number): Waypoint => ({
  forecast_hour: hour,
  lat,
  lon,
  wind_kt: 90,
  pressure_hpa: 960,
  uncertainty_radius_km: radius,
  dist2land_km: 300,
});

// A waypoint outside the shipped coastline's coverage carries no distance at all.
const uncovered = (hour: number, lat: number, lon: number): Waypoint => ({
  ...waypoint(hour, lat, lon, 50),
  dist2land_km: null,
});

// A due-north track out of the central Bay: the cone's two edges must run east and west.
const NORTHWARD: Waypoint[] = [
  waypoint(6, 15.0, 87.0, 30),
  waypoint(12, 16.0, 87.0, 60),
  waypoint(24, 18.0, 87.0, 140),
  waypoint(48, 21.0, 87.0, 300),
];

test("km converts to degrees of latitude at the standard spherical rate", () => {
  const { dLat } = kmToDegrees(111.195, 0);
  assert.ok(Math.abs(dLat - 1) < 0.01, `expected about 1 degree, got ${dLat}`);
});

test("a degree of longitude costs fewer km as latitude rises", () => {
  const equator = kmToDegrees(100, 0);
  const sixty = kmToDegrees(100, 60);
  assert.ok(Math.abs(equator.dLon - equator.dLat) < 1e-9);
  assert.ok(Math.abs(sixty.dLon - 2 * sixty.dLat) < 0.01, `expected double, got ${sixty.dLon / sixty.dLat}`);
});

test("longitude scaling stays finite near the pole instead of dividing by zero", () => {
  assert.ok(Number.isFinite(kmToDegrees(100, 90).dLon));
});

// Ring layout: the left edge runs origin-to-last as indices 0..N, the right edge comes
// back as N+1..2N+1, and index 2N+2 closes it. So anchor i pairs index i with 2N+1-i.
const edges = (ring: [number, number][], anchor: number, waypoints = NORTHWARD.length) =>
  [ring[anchor], ring[2 * waypoints + 1 - anchor]] as const;

test("the cone is a closed ring with one point per side of every waypoint plus the origin", () => {
  const ring = conePolygon([14.0, 87.0], NORTHWARD);
  assert.equal(ring.length, 2 * (NORTHWARD.length + 1) + 1);
  assert.deepEqual(ring[0], ring[ring.length - 1]);
});

test("the cone widens with the uncertainty radius, not with lead time alone", () => {
  const narrow = conePolygon([14.0, 87.0], NORTHWARD.map((w) => ({ ...w, uncertainty_radius_km: 10 })));
  const wide = conePolygon([14.0, 87.0], NORTHWARD.map((w) => ({ ...w, uncertainty_radius_km: 200 })));
  const spread = (ring: [number, number][]) => {
    const [left, right] = edges(ring, NORTHWARD.length);
    return Math.abs(left[1] - right[1]);
  };
  assert.ok(spread(wide) > spread(narrow) * 5, `wide ${spread(wide)} vs narrow ${spread(narrow)}`);
});

test("the cone widens along the track, so the far end is broader than the near end", () => {
  const ring = conePolygon([14.0, 87.0], NORTHWARD);
  const [nearLeft, nearRight] = edges(ring, 1);
  const [farLeft, farRight] = edges(ring, NORTHWARD.length);
  const near = Math.abs(nearLeft[1] - nearRight[1]);
  const far = Math.abs(farLeft[1] - farRight[1]);
  assert.ok(far > near, `far ${far} should exceed near ${near}`);
});

test("a due-north track produces east and west cone edges at the same latitude", () => {
  const ring = conePolygon([14.0, 87.0], NORTHWARD);
  const [left, right] = edges(ring, NORTHWARD.length);
  assert.ok(Math.abs(left[0] - right[0]) < 0.3, "the two edges should sit beside the waypoint");
  assert.ok(left[1] !== right[1], "the two edges should fall on opposite sides");
});

test("a single waypoint still yields a usable ring", () => {
  const ring = conePolygon([14.0, 87.0], [waypoint(6, 15, 87, 40)]);
  assert.equal(ring.length, 2 * 2 + 1);
  assert.ok(ring.every(([lat, lon]) => Number.isFinite(lat) && Number.isFinite(lon)));
});

test("no waypoints yields no ring rather than a degenerate shape", () => {
  assert.deepEqual(conePolygon([14.0, 87.0], []), []);
});

test("track points are ordered oldest to newest by their timestamps", () => {
  const points = [
    { lat: 16, lon: 87, time: "2026-10-22T12:00:00.000Z" },
    { lat: 14, lon: 86, time: "2026-10-22T00:00:00.000Z" },
    { lat: 15, lon: 86.5, time: "2026-10-22T06:00:00.000Z" },
  ];
  assert.deepEqual(trackLatLngs(points), [[14, 86], [15, 86.5], [16, 87]]);
});

test("track points without timestamps keep the order they arrived in", () => {
  const points = [
    { lat: 14, lon: 86, time: null },
    { lat: 15, lon: 86.5, time: null },
  ];
  assert.deepEqual(trackLatLngs(points), [[14, 86], [15, 86.5]]);
});

test("observations take the last four fixes and put the reported wind on the latest", () => {
  const points = Array.from({ length: 6 }, (_, i) => ({
    lat: 12 + i,
    lon: 86 + i * 0.2,
    time: `2026-10-2${i + 1}T00:00:00.000Z`,
  }));
  const observations = observationsFromTrack(points, 110);
  assert.equal(observations.length, 4);
  assert.deepEqual(
    observations.map((o) => o.lat),
    [14, 15, 16, 17],
  );
  assert.equal(observations[observations.length - 1].wind_kt, 110);
  assert.equal(observations[0].wind_kt, undefined);
});

test("observations survive a track with no reported wind", () => {
  const observations = observationsFromTrack([{ lat: 14, lon: 86, time: null }], null);
  assert.equal(observations.length, 1);
  assert.equal(observations[0].wind_kt, undefined);
});

test("the highlighted leads are the ones the dashboard labels", () => {
  assert.deepEqual(WAYPOINT_HIGHLIGHTS, [12, 24, 48]);
});

test("a waypoint with no coastline coverage still draws a cone", () => {
  const ring = conePolygon([14.0, 87.0], [uncovered(6, 15, 87), uncovered(12, 16, 87)]);
  assert.equal(ring.length, 2 * 3 + 1);
  assert.ok(ring.every(([lat, lon]) => Number.isFinite(lat) && Number.isFinite(lon)));
});

// --- regressions from the final review ---

test("a stationary forecast still draws a cone with area", () => {
  // One observation and no checkpoint gives zero motion at every lead. The panel still
  // quotes hundreds of km of uncertainty, so a zero-area ring would hide it.
  const still: Waypoint[] = [6, 12, 24, 48].map((hour) => waypoint(hour, 14.1, 86.4, 30 * (hour / 6)));
  const ring = conePolygon([14.1, 86.4], still);
  const lons = ring.map(([, lon]) => lon);
  const lats = ring.map(([lat]) => lat);
  const spread = Math.max(...lons) - Math.min(...lons) + (Math.max(...lats) - Math.min(...lats));
  assert.ok(spread > 0.5, `expected a visible cone, got a spread of ${spread}`);
});

test("mixed timestamped and untimed points sort deterministically", () => {
  const points = Array.from({ length: 14 }, (_, i) => ({
    lat: 10 + i,
    lon: 80,
    time: i % 2 === 0 ? `2026-10-${String(10 + i).padStart(2, "0")}T00:00:00.000Z` : null,
  }));
  const first = trackLatLngs(points);
  const second = trackLatLngs([...points]);
  assert.deepEqual(first, second);
  // Every timestamped point keeps ascending time order.
  const timed = points.filter((p) => p.time).map((p) => p.lat);
  const orderInResult = first.map(([lat]) => lat).filter((lat) => timed.includes(lat));
  assert.deepEqual(orderInResult, timed);
});

test("positions name the hemisphere they are actually in", () => {
  assert.equal(formatPosition(14.1, 86.4), "14.1\u00b0N 86.4\u00b0E");
  assert.equal(formatPosition(-18.0, -170.0), "18.0\u00b0S 170.0\u00b0W");
  assert.equal(formatPosition(0, 0), "0.0\u00b0N 0.0\u00b0E");
});

test("a pressure change of zero is not described as deepening", () => {
  assert.match(describePressureChange(0), /no net change/i);
  assert.match(describePressureChange(12), /filling|weaken/i);
  assert.match(describePressureChange(-12), /deepening/i);
});

test("a landfall already in progress is not described as being in zero hours", () => {
  const already = describeLandfall({
    projected: true,
    location: "West Bengal / Sundarbans coast",
    lat: 21.6,
    lon: 88.3,
    eta_hours: 0,
    note: "The storm centre is already at the coast at the latest observation.",
    threshold_km: 35,
  });
  assert.doesNotMatch(already, /0 hours/);
  assert.match(already, /already at the coast/i);
});

test("a projected landfall still reports its location and timing", () => {
  const soon = describeLandfall({
    projected: true,
    location: "Odisha coast near Puri",
    lat: 19.8,
    lon: 85.8,
    eta_hours: 13,
    note: "Projected crossing: ...",
    threshold_km: 35,
  });
  assert.match(soon, /Odisha coast near Puri/);
  assert.match(soon, /13 hours/);
});

test("no projected landfall shows the engine's own explanation", () => {
  const none = describeLandfall({
    projected: false,
    location: null,
    lat: null,
    lon: null,
    eta_hours: null,
    note: "No coastal crossing within 48 h. The forecast track stays 38 km offshore at its closest.",
    threshold_km: 35,
  });
  assert.match(none, /38 km offshore/);
});

test("a storm outside the North Indian Ocean is recognised as out of basin", () => {
  assert.equal(isOutsideHomeBasin("Outside the North Indian Ocean"), true);
  assert.equal(isOutsideHomeBasin("Bay of Bengal"), false);
  assert.equal(isOutsideHomeBasin("Arabian Sea"), false);
  assert.equal(isOutsideHomeBasin("North Indian Ocean"), false);
});
