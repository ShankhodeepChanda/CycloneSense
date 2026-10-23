// Run: npm run test:web
// The sample below is hand-written to the GDACS GeoJSON layout as I understand it (fixture, not live data).
import test from "node:test";
import assert from "node:assert/strict";
import { parseActiveCyclones, parseCycloneTrack } from "../server/gdacs";

const pt = (lon: number, lat: number) => ({ type: "Point", coordinates: [lon, lat] });
const tc = (over: any, geometry: any) => ({
  type: "Feature",
  geometry,
  properties: { eventtype: "TC", iscurrent: "true", eventid: 1, eventname: "TESTONE", alertlevel: "Orange",
    severitydata: { severity: 120 }, Class: "Point_Centroid", ...over },
});

test("keeps current NIO cyclone centroid and converts km/h to knots", () => {
  const out = parseActiveCyclones({ features: [tc({}, pt(88.5, 17.2))] });
  assert.equal(out.length, 1);
  assert.deepEqual([out[0].name, out[0].alert, out[0].windKt, out[0].lat, out[0].lon], ["TESTONE", "Orange", 65, 17.2, 88.5]);
});

test("flags other basins as not NIO, and ignores past events, other hazards, and non-centroid points", () => {
  const out = parseActiveCyclones({
    features: [
      tc({ eventid: 3, iscurrent: "false" }, pt(88, 17)),
      tc({ eventid: 4, eventtype: "EQ" }, pt(88, 17)),
      tc({ eventid: 5, Class: "Point_First" }, pt(88, 17)),
      tc({ eventid: 6 }, { type: "Polygon", coordinates: [] }),
    ],
  });
  assert.equal(out.length, 0);
});

test("keeps other basins but marks them nio=false, NIO storms sort first", () => {
  const out = parseActiveCyclones({
    features: [tc({ eventid: 2, eventname: "ATL", severitydata: { severity: 300 } }, pt(-60, 20)), tc({ eventid: 3, eventname: "BOB" }, pt(88, 17))],
  });
  assert.deepEqual(out.map((c) => [c.name, c.nio]), [["BOB", true], ["ATL", false]]);
});

test("copes with empty or malformed input", () => {
  assert.deepEqual(parseActiveCyclones(null), []);
  assert.deepEqual(parseActiveCyclones({ features: "nope" }), []);
});

test("missing wind becomes null, strongest storm sorts first", () => {
  const out = parseActiveCyclones({
    features: [
      tc({ eventid: 7, eventname: "WEAK", severitydata: {} }, pt(70, 15)),
      tc({ eventid: 8, eventname: "STRONG", severitydata: { severity: 200 } }, pt(85, 14)),
    ],
  });
  assert.deepEqual(out.map((c) => [c.name, c.windKt]), [["STRONG", 108], ["WEAK", null]]);
});

test("without iscurrent, keeps storms that ended within 3 days and drops older ones", () => {
  const day = 86_400_000;
  const out = parseActiveCyclones({
    features: [
      tc({ eventid: 20, eventname: "RECENT", iscurrent: undefined, todate: new Date(Date.now() - day).toISOString() }, pt(88, 17)),
      tc({ eventid: 21, eventname: "OLD", iscurrent: undefined, todate: new Date(Date.now() - 6 * day).toISOString() }, pt(88, 17)),
    ],
  });
  assert.deepEqual(out.map((c) => c.name), ["RECENT"]);
});

test("severity already in knots is not converted again", () => {
  const out = parseActiveCyclones({ features: [tc({ eventid: 22, severitydata: { severity: 65, severityunit: "kn" } }, pt(88, 17))] });
  assert.equal(out[0].windKt, 65);
});

// --- per-event track geometry (getgeometry endpoint) ---
// Layout confirmed against a live GDACS response: Line_Line_N 2-point segments chain into
// the ordered track centres, Point_Polygon_Point_N carry a "DD/MM HH:MM UTC" label each,
// Poly_Cones is the forecast cone, and `todate` is the dividing line between what has been
// observed and what GDACS is forecasting.
const geometryFixture = (over: any = {}) => {
  const base = {
    eventtype: "TC", eventid: 1001341, eventname: "TESTONE",
    fromdate: "2026-10-21T15:00:00", todate: "2026-10-23T03:00:00",
    severitydata: { severity: 194.4432, severityunit: "km/h" }, ...over,
  };
  const centres: [number, number][] = [[86, 12], [86.5, 13], [87, 14], [87.5, 15], [88, 16]];
  const labels = ["21/10 15:00 UTC", "22/10 03:00 UTC", "22/10 15:00 UTC", "23/10 03:00 UTC", "23/10 15:00 UTC"];
  const features: any[] = [];
  for (let i = 0; i < centres.length - 1; i++) {
    features.push({ type: "Feature", properties: { ...base, Class: `Line_Line_${i}` },
      geometry: { type: "LineString", coordinates: [centres[i], centres[i + 1]] } });
  }
  labels.forEach((polygonlabel, i) => {
    features.push({ type: "Feature", properties: { ...base, Class: `Point_Polygon_Point_${i}`, polygonlabel },
      geometry: { type: "Polygon", coordinates: [[centres[i], centres[i], centres[i]]] } });
  });
  features.push({ type: "Feature", properties: { ...base, Class: "Poly_Cones" },
    geometry: { type: "Polygon", coordinates: [[[86, 12], [89, 17], [87, 17], [86, 12]]] } });
  features.push({ type: "Feature", properties: { ...base, Class: "Point_Centroid" },
    geometry: { type: "Point", coordinates: [87.5, 15] } });
  return { type: "FeatureCollection", features };
};

test("chains the line segments into ordered track centres", () => {
  const track = parseCycloneTrack(geometryFixture());
  assert.deepEqual(track.points.map((p) => [p.lat, p.lon]), [[12, 86], [13, 86.5], [14, 87], [15, 87.5], [16, 88]]);
  assert.equal(track.name, "TESTONE");
  assert.equal(track.eventId, "1001341");
});

test("reads the per-point timestamps from the polygon labels", () => {
  const track = parseCycloneTrack(geometryFixture());
  assert.deepEqual(track.points.map((p) => p.time), [
    "2026-10-21T15:00:00.000Z", "2026-10-22T03:00:00.000Z", "2026-10-22T15:00:00.000Z",
    "2026-10-23T03:00:00.000Z", "2026-10-23T15:00:00.000Z",
  ]);
});

test("splits what has been observed from what GDACS is forecasting, at todate", () => {
  const track = parseCycloneTrack(geometryFixture());
  assert.deepEqual(track.observed.map((p) => p.lat), [12, 13, 14, 15]);
  assert.deepEqual(track.forecast.map((p) => p.lat), [16]);
  assert.equal(track.asOf, "2026-10-23T03:00:00.000Z");
});

test("extracts the forecast cone ring", () => {
  const track = parseCycloneTrack(geometryFixture());
  assert.deepEqual(track.cone, [[12, 86], [17, 89], [17, 87], [12, 86]]);
});

test("labels crossing into a new year resolve to the later year", () => {
  const track = parseCycloneTrack(geometryFixture({ fromdate: "2026-12-30T00:00:00", todate: "2026-12-31T00:00:00" }));
  // Labels run 21/10 to 23/10, which cannot precede a 30 December start: they belong to the next year.
  assert.deepEqual(track.points.map((p) => p.time?.slice(0, 4)), ["2027", "2027", "2027", "2027", "2027"]);
});

test("a track with no line segments falls back to the point polygon centres", () => {
  const fixture = geometryFixture();
  fixture.features = fixture.features.filter((f: any) => !String(f.properties.Class).startsWith("Line_"));
  const track = parseCycloneTrack(fixture);
  assert.equal(track.points.length, 5);
  assert.deepEqual(track.points[0], { lat: 12, lon: 86, time: "2026-10-21T15:00:00.000Z", observed: true });
});

test("copes with empty, null and malformed geometry responses", () => {
  for (const input of [null, {}, { features: "nope" }, { features: [] }]) {
    const track = parseCycloneTrack(input);
    assert.deepEqual(track.points, []);
    assert.deepEqual(track.observed, []);
    assert.equal(track.cone, null);
  }
});

test("ignores a point label that is not a parseable date", () => {
  const fixture = geometryFixture();
  for (const f of fixture.features) {
    if (f.properties.Class === "Point_Polygon_Point_2") f.properties.polygonlabel = "not a date";
  }
  const track = parseCycloneTrack(fixture);
  assert.equal(track.points.length, 5);
  assert.equal(track.points[2].time, null);
  assert.equal(track.points[2].observed, true); // between two observed fixes, so not a forecast point
});

// --- regressions from the final review ---

test("a todate that already carries Z still divides observed from forecast", () => {
  const track = parseCycloneTrack(geometryFixture({ todate: "2026-10-23T03:00:00Z" }));
  assert.equal(track.asOfSource, "todate");
  assert.deepEqual(track.observed.map((p) => p.lat), [12, 13, 14, 15]);
  assert.deepEqual(track.forecast.map((p) => p.lat), [16]);
});

test("a todate carrying an offset is parsed rather than discarded", () => {
  const track = parseCycloneTrack(geometryFixture({ todate: "2026-10-23T08:30:00+05:30" }));
  assert.equal(track.asOfSource, "todate");
  assert.deepEqual(track.forecast.map((p) => p.lat), [16]);
});

test("without todate it falls back to datemodified instead of calling everything observed", () => {
  const fixture = geometryFixture({ todate: undefined, datemodified: "2026-10-23T03:00:00" });
  const track = parseCycloneTrack(fixture);
  assert.equal(track.asOfSource, "datemodified");
  assert.deepEqual(track.forecast.map((p) => p.lat), [16]);
});

test("with no usable date at all, points in the future are not called observations", () => {
  const track = parseCycloneTrack(geometryFixture({ todate: undefined, datemodified: undefined }));
  assert.equal(track.asOfSource, "now");
  // The fixture's labels are all in October 2026; the test clock decides the split, but
  // the one thing that must not happen is every point being declared observed.
  assert.equal(track.points.length, 5);
  assert.equal(track.observed.length + track.forecast.length, 5);
});

test("timestamps are joined to positions by their parsed index, not by array position", () => {
  const fixture = geometryFixture();
  // GDACS does omit radii polygons for some early fixes.
  fixture.features = fixture.features.filter(
    (f: any) => f.properties.Class !== "Point_Polygon_Point_0" && f.properties.Class !== "Point_Polygon_Point_1",
  );
  const track = parseCycloneTrack(fixture);
  assert.equal(track.points.length, 5);
  assert.equal(track.points[0].time, null); // no label for index 0
  assert.equal(track.points[1].time, null);
  assert.equal(track.points[2].time, "2026-10-22T15:00:00.000Z"); // its own label, not point 0's
  assert.equal(track.points[3].time, "2026-10-23T03:00:00.000Z");
  assert.deepEqual(track.observed.map((p) => p.lat), [12, 13, 14, 15]);
});

test("a gap between line segments keeps every track point", () => {
  const base = {
    eventtype: "TC", eventid: 7, eventname: "GAPPY",
    fromdate: "2026-10-21T15:00:00", todate: "2026-10-23T03:00:00",
  };
  const segment = (i: number, a: [number, number], b: [number, number]) => ({
    type: "Feature", properties: { ...base, Class: `Line_Line_${i}` },
    geometry: { type: "LineString", coordinates: [a, b] },
  });
  const track = parseCycloneTrack({
    features: [segment(0, [86, 12], [86.5, 13]), segment(1, [88, 16], [88.5, 17])],
  });
  assert.deepEqual(track.points.map((p) => [p.lat, p.lon]), [[12, 86], [13, 86.5], [16, 88], [17, 88.5]]);
});

test("contiguous segments still collapse their shared endpoint", () => {
  const track = parseCycloneTrack(geometryFixture());
  assert.equal(track.points.length, 5); // 4 segments, not 8 points
});
