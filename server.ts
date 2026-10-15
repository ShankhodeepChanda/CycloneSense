import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { getActiveCyclones } from "./server/gdacs";

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  const health = (_req: express.Request, res: express.Response) => res.json({ status: "ok", service: "cyclonesense" });
  app.get("/health", health);
  app.get("/api/health", health);

  app.get("/api/active-cyclones", async (_req, res) => {
    res.set("Cache-Control", "public, max-age=300");
    res.json(await getActiveCyclones());
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
