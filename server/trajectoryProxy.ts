import express, { type Application, type NextFunction, type Request, type Response } from "express";

/** Largest trajectory request worth forwarding: a storm's observed track, not a dataset. */
const MAX_BODY = "256kb";

/**
 * Forward trajectory forecast requests to the FastAPI engine from the same origin.
 *
 * The browser talks only to this server, so the Python API needs no CORS exposure and no
 * public port. Upstream status codes pass through unchanged, because 422 ("these
 * observations cannot support a forecast") is a real answer the UI shows to the user.
 */
export function registerTrajectoryProxy(
  app: Application,
  apiBase = process.env.PYTHON_API_URL || "http://127.0.0.1:8000",
) {
  const path = "/api/predict/trajectory";

  app.post(
    path,
    (req: Request, res: Response, next: NextFunction) => {
      const contentType = String(req.headers["content-type"] ?? "").toLowerCase();
      if (!contentType.includes("application/json")) {
        res.status(415).json({ detail: "Send the observations as application/json." });
        return;
      }
      next();
    },
    express.json({ limit: MAX_BODY }),
    async (req: Request, res: Response) => {
      try {
        const upstream = await fetch(new URL("/predict/trajectory", apiBase), {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(req.body ?? {}),
          signal: AbortSignal.timeout(60_000),
        });
        res.status(upstream.status);
        const responseType = upstream.headers.get("content-type");
        if (responseType) res.setHeader("content-type", responseType);
        res.send(Buffer.from(await upstream.arrayBuffer()));
      } catch (error) {
        console.error("Trajectory forecast proxy failed", error);
        if (!res.headersSent) {
          res.status(503).json({
            detail: "The trajectory forecast service is unavailable. Start the Python API and try again.",
          });
        }
      }
    },
    // Body-parser failures land here: report them instead of letting Express emit HTML.
    (error: any, _req: Request, res: Response, next: NextFunction) => {
      if (res.headersSent) return next(error);
      if (error?.type === "entity.too.large") {
        res.status(413).json({ detail: "That request is too large. Send the storm's recent fixes, not a dataset." });
        return;
      }
      res.status(400).json({ detail: "Could not read the observations as JSON." });
    },
  );
}
