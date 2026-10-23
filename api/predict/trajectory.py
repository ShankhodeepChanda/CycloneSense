"""Vercel serverless function: POST /api/predict/trajectory.

The same forecast the Express proxy serves locally, but run in-process instead of being
forwarded to a FastAPI container. PyTorch is 534 MB against Vercel's 250 MB function
limit, so this path reads the exported numpy weights
(``models/bob_trajectory_gru.npz``) through ``app.trajectory``; the engine prefers them
over the torch checkpoint wherever both exist, and
``tests/test_numpy_forecaster.py`` pins the two runtimes to the same outputs.

FastAPI is deliberately not imported: it would add ~35 MB for one route, so the few
request checks it would have done are done here instead.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# The bundle root holds app/ and ml/ (see includeFiles in vercel.json); make them importable
# regardless of the working directory Vercel chooses.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.trajectory import engine  # noqa: E402  (must follow the sys.path setup)

MAX_BODY_BYTES = 256 * 1024
MAX_OBSERVATIONS = 400


def _validate(payload: object) -> tuple[str, list[dict]]:
    """The checks Pydantic did server-side, kept because FastAPI is too heavy to bundle."""
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object with cyclone_id and observations.")

    cyclone_id = payload.get("cyclone_id")
    if not isinstance(cyclone_id, str) or not cyclone_id.strip():
        raise ValueError("cyclone_id must be a non-empty string.")
    if len(cyclone_id) > 64:
        raise ValueError("cyclone_id must be 64 characters or fewer.")

    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty list, earliest fix first.")
    if len(observations) > MAX_OBSERVATIONS:
        raise ValueError(f"Send at most {MAX_OBSERVATIONS} observations.")

    cleaned: list[dict] = []
    for index, raw in enumerate(observations):
        if not isinstance(raw, dict):
            raise ValueError(f"Observation {index} is not an object.")
        try:
            lat = float(raw["lat"])
            lon = float(raw["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Observation {index} needs numeric lat and lon.") from exc
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"Observation {index}: lat must be between -90 and 90.")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"Observation {index}: lon must be between -180 and 180.")

        fix: dict = {"lat": lat, "lon": lon}
        for field, low, high in (("wind_kt", 0.0, 300.0), ("pressure_hpa", 800.0, 1050.0)):
            value = raw.get(field)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Observation {index}: {field} must be a number.") from exc
            if not low <= number <= high:
                raise ValueError(f"Observation {index}: {field} must be between {low:.0f} and {high:.0f}.")
            fix[field] = number
        timestamp = raw.get("timestamp")
        if timestamp is not None:
            fix["timestamp"] = str(timestamp)[:64]
        cleaned.append(fix)

    return cyclone_id.strip(), cleaned


class handler(BaseHTTPRequestHandler):  # noqa: N801  (Vercel requires this name)
    def _send(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802  (http.server's interface)
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            self._send(400, {"detail": "A numeric content-length is required."})
            return
        if length <= 0:
            self._send(400, {"detail": "Send the observations as a JSON body."})
            return
        if length > MAX_BODY_BYTES:
            self._send(413, {"detail": "That request is too large. Send the storm's recent fixes, not a dataset."})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"detail": "Could not read the observations as JSON."})
            return

        try:
            cyclone_id, observations = _validate(payload)
        except ValueError as exc:
            self._send(422, {"detail": str(exc)})
            return

        try:
            self._send(200, engine.forecast(cyclone_id, observations))
        except ValueError as exc:
            # The engine refuses inputs it cannot forecast from honestly, such as a
            # position with no wind and no pressure anywhere in the sequence.
            self._send(422, {"detail": str(exc)})
        except Exception:
            self.log_error("trajectory forecast failed for %s", cyclone_id)
            self._send(503, {"detail": "The trajectory forecast service is unavailable."})

    def do_GET(self) -> None:  # noqa: N802
        self._send(405, {"detail": "Use POST with a JSON body of observations."})

    def log_message(self, fmt: str, *args) -> None:
        """Log to stderr without the default timestamped client-address prefix."""
        sys.stderr.write("trajectory: " + (fmt % args) + "\n")
