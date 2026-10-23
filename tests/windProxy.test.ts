import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { test } from "node:test";
import express from "express";
import { registerWindPredictionProxy } from "../server/windProxy";

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") return reject(new Error("No test server address"));
      resolve(address.port);
    });
  });
}

function close(server: Server): Promise<void> {
  return new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

test("same-origin proxy forwards multipart wind uploads and preserves the API response", async () => {
  let receivedType = "";
  let receivedBody = "";
  const upstream = createServer((request, response) => {
    receivedType = request.headers["content-type"] || "";
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => {
      receivedBody = Buffer.concat(chunks).toString("latin1");
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ raw_prediction_kt: -4, application_prediction_kt: 0 }));
    });
  });
  const upstreamPort = await listen(upstream);
  const web = express();
  registerWindPredictionProxy(web, `http://127.0.0.1:${upstreamPort}`);
  const webServer = web.listen(0, "127.0.0.1");
  const webPort = await new Promise<number>((resolve, reject) => {
    webServer.once("error", reject);
    webServer.once("listening", () => {
      const address = webServer.address();
      if (!address || typeof address === "string") return reject(new Error("No web test address"));
      resolve(address.port);
    });
  });
  try {
    const form = new FormData();
    form.append("file", new Blob(["test-image-bytes"], { type: "image/png" }), "sample.png");
    const result = await fetch(`http://127.0.0.1:${webPort}/api/predict/wind`, { method: "POST", body: form });
    assert.equal(result.status, 200);
    assert.deepEqual(await result.json(), { raw_prediction_kt: -4, application_prediction_kt: 0 });
    assert.match(receivedType, /^multipart\/form-data; boundary=/);
    assert.match(receivedBody, /name="file"; filename="sample.png"/);
    assert.match(receivedBody, /test-image-bytes/);

    const rejected = await fetch(`http://127.0.0.1:${webPort}/api/predict/wind`, {
      method: "POST", headers: { "content-type": "application/json" }, body: "{}",
    });
    assert.equal(rejected.status, 415);
  } finally {
    await Promise.all([close(webServer), close(upstream)]);
  }
});
