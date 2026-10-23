// Run: npm run test:web
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { test } from "node:test";
import express from "express";
import { registerTrajectoryProxy } from "../server/trajectoryProxy";

function port(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    const ready = () => {
      const address = server.address();
      if (!address || typeof address === "string") return reject(new Error("No test server address"));
      resolve(address.port);
    };
    server.listening ? ready() : server.once("listening", ready);
  });
}

function close(server: Server): Promise<void> {
  return new Promise((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
}

const OBSERVATIONS = [
  { lat: 12.5, lon: 86.4, wind_kt: 120, pressure_hpa: 932 },
  { lat: 14.1, lon: 86.4, wind_kt: 145, pressure_hpa: 901 },
];

async function withProxy(
  handler: Parameters<typeof createServer>[1] | null,
  run: (base: string) => Promise<void>,
) {
  const upstream = handler ? createServer(handler) : null;
  let apiBase = "http://127.0.0.1:1"; // nothing listening: exercises the unreachable path
  if (upstream) {
    upstream.listen(0, "127.0.0.1");
    apiBase = `http://127.0.0.1:${await port(upstream)}`;
  }
  const web = express();
  registerTrajectoryProxy(web, apiBase);
  const webServer = web.listen(0, "127.0.0.1");
  const webPort = await port(webServer);
  try {
    await run(`http://127.0.0.1:${webPort}`);
  } finally {
    await Promise.all([close(webServer), ...(upstream ? [close(upstream)] : [])]);
  }
}

test("forwards the forecast request to FastAPI and returns its body", async () => {
  let receivedBody = "";
  let receivedPath = "";
  await withProxy(
    (request, response) => {
      receivedPath = request.url ?? "";
      const chunks: Buffer[] = [];
      request.on("data", (chunk: Buffer) => chunks.push(chunk));
      request.on("end", () => {
        receivedBody = Buffer.concat(chunks).toString("utf8");
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({ cyclone_id: "AMPHAN", waypoints: [{ forecast_hour: 6 }] }));
      });
    },
    async (base) => {
      const result = await fetch(`${base}/api/predict/trajectory`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ cyclone_id: "AMPHAN", observations: OBSERVATIONS }),
      });
      assert.equal(result.status, 200);
      assert.deepEqual(await result.json(), { cyclone_id: "AMPHAN", waypoints: [{ forecast_hour: 6 }] });
      assert.equal(receivedPath, "/predict/trajectory");
      assert.deepEqual(JSON.parse(receivedBody).observations, OBSERVATIONS);
    },
  );
});

test("passes an upstream validation failure through with its status and detail", async () => {
  await withProxy(
    (_request, response) => {
      response.writeHead(422, { "content-type": "application/json" });
      response.end(JSON.stringify({ detail: "At least one observation must carry wind_kt or pressure_hpa." }));
    },
    async (base) => {
      const result = await fetch(`${base}/api/predict/trajectory`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ cyclone_id: "X", observations: [{ lat: 15, lon: 88 }] }),
      });
      assert.equal(result.status, 422);
      assert.match((await result.json()).detail, /wind_kt or pressure_hpa/);
    },
  );
});

test("answers 503 with a readable detail when the Python API is not running", async () => {
  await withProxy(null, async (base) => {
    const result = await fetch(`${base}/api/predict/trajectory`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ cyclone_id: "AMPHAN", observations: OBSERVATIONS }),
    });
    assert.equal(result.status, 503);
    assert.match((await result.json()).detail, /Python API|unavailable/i);
  });
});

test("rejects a request that is not JSON before touching the upstream", async () => {
  let upstreamWasCalled = false;
  await withProxy(
    (_request, response) => {
      upstreamWasCalled = true;
      response.writeHead(200, { "content-type": "application/json" });
      response.end("{}");
    },
    async (base) => {
      const result = await fetch(`${base}/api/predict/trajectory`, {
        method: "POST",
        headers: { "content-type": "text/plain" },
        body: "cyclone_id=AMPHAN",
      });
      assert.equal(result.status, 415);
      assert.equal(upstreamWasCalled, false);
    },
  );
});

test("rejects a body larger than the request limit", async () => {
  await withProxy(
    (_request, response) => {
      response.writeHead(200, { "content-type": "application/json" });
      response.end("{}");
    },
    async (base) => {
      const observations = Array.from({ length: 200_000 }, () => ({ lat: 12.5, lon: 86.4, wind_kt: 120 }));
      const result = await fetch(`${base}/api/predict/trajectory`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ cyclone_id: "HUGE", observations }),
      });
      assert.equal(result.status, 413);
    },
  );
});
