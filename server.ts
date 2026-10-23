import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { getActiveCyclones, getCycloneTrack } from "./server/gdacs";
import { registerWindPredictionProxy } from "./server/windProxy";
import { registerTrajectoryProxy } from "./server/trajectoryProxy";

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  const health = (_req: express.Request, res: express.Response) => res.json({ status: "ok", service: "cyclonesense" });
  app.get("/health", health);
  app.get("/api/health", health);

  registerWindPredictionProxy(app);
  registerTrajectoryProxy(app);

  app.get("/api/active-cyclones", async (_req, res) => {
    res.set("Cache-Control", "public, max-age=300");
    res.json(await getActiveCyclones());
  });

  // One storm's observed track and GDACS's own cone, for the map's history polyline.
  app.get("/api/cyclone-track", async (req, res) => {
    const eventId = String(req.query.eventId ?? "").trim();
    if (!/^[0-9]{1,12}$/.test(eventId)) {
      res.status(400).json({ ok: false, error: "Pass eventId as the numeric GDACS event id." });
      return;
    }
    const episode = String(req.query.episodeId ?? "").trim();
    res.set("Cache-Control", "public, max-age=300");
    res.json(await getCycloneTrack(eventId, /^[0-9]{1,12}$/.test(episode) ? episode : null));
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({ server: { middlewareMode: true }, appType: "spa" });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => res.sendFile(path.join(distPath, "index.html")));
  }

  app.listen(PORT, "0.0.0.0", () => console.log(`CycloneSense running on http://0.0.0.0:${PORT}`));
}

startServer();
