// Run: npm run test:web
// The sample below is hand-written to the GDACS GeoJSON layout as I understand it (fixture, not live data).
import test from "node:test";
import assert from "node:assert/strict";
import { parseActiveCyclones } from "../server/gdacs";

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

test("ignores other basins, past events, other hazards, and non-centroid points", () => {
  const out = parseActiveCyclones({
    features: [
      tc({ eventid: 2 }, pt(-60, 20)), // Atlantic
      tc({ eventid: 3, iscurrent: "false" }, pt(88, 17)),
      tc({ eventid: 4, eventtype: "EQ" }, pt(88, 17)),
      tc({ eventid: 5, Class: "Point_First" }, pt(88, 17)),
      tc({ eventid: 6 }, { type: "Polygon", coordinates: [] }),
    ],
  });
  assert.equal(out.length, 0);
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
