// Vercel serverless function; the Express server (server.ts) serves the same route for Docker.
import { getCycloneTrack } from "../server/gdacs.js";

const NUMERIC_ID = /^[0-9]{1,12}$/;

export default async function handler(req: any, res: any) {
  const eventId = String(req.query?.eventId ?? "").trim();
  if (!NUMERIC_ID.test(eventId)) {
    res.status(400).json({ ok: false, error: "Pass eventId as the numeric GDACS event id." });
    return;
  }
  const episode = String(req.query?.episodeId ?? "").trim();
  res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=600");
  res.status(200).json(await getCycloneTrack(eventId, NUMERIC_ID.test(episode) ? episode : null));
}
