// Vercel serverless function: POST /api/predict/wind.
//
// Model A is a ResNet-18 and needs PyTorch, which is 534 MB against Vercel's 250 MB
// function limit, so the wind estimator cannot run on this deployment. Its checkpoint
// is also absent from the repository.
//
// This exists so the browser gets that explanation instead of a bare 404 from a missing
// route, which the UI would otherwise report as "could not process this image".

export default async function handler(_req: unknown, res: any) {
  res.status(503).json({
    detail:
      "Wind estimation from imagery is not available on this deployment: Model A needs PyTorch, " +
      "which exceeds the serverless function size limit. Run the Python service locally " +
      "(python run_server.py) with the Model A checkpoint installed to use it.",
  });
}
