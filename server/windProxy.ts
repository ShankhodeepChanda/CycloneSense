import type { Application } from "express";

/** Forward multipart wind uploads to FastAPI without buffering the image in Node. */
export function registerWindPredictionProxy(
  app: Application,
  apiBase = process.env.PYTHON_API_URL || "http://127.0.0.1:8000",
) {
  app.post("/api/predict/wind", async (req, res) => {
    const contentType = req.headers["content-type"];
    if (!contentType?.toLowerCase().startsWith("multipart/form-data;")) {
      res.status(415).json({ detail: "Please upload an image using multipart form data." });
      return;
    }
    try {
      const upstream = await fetch(new URL("/predict/wind", apiBase), {
        method: "POST",
        headers: { "content-type": contentType },
        body: req as any,
        duplex: "half",
        signal: AbortSignal.timeout(120_000),
      } as any);
      res.status(upstream.status);
      const responseType = upstream.headers.get("content-type");
      if (responseType) res.setHeader("content-type", responseType);
      res.send(Buffer.from(await upstream.arrayBuffer()));
    } catch (error) {
      console.error("Wind inference API proxy failed", error);
      if (!res.headersSent) {
        res.status(503).json({ detail: "The wind estimation service is unavailable. Start the Python API and try again." });
      }
    }
  });
}
