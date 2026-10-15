// Vercel serverless function; the Express server (server.ts) serves the same route for Docker.
import { getActiveCyclones } from "../server/gdacs";

export default async function handler(_req: unknown, res: any) {
  res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=600");
  res.status(200).json(await getActiveCyclones());
}
