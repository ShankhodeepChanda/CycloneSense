# Deployment

Two supported targets. **Vercel** gives one shareable link with the forecast working and
is the easiest path; **Cloud Run** is for running the full PyTorch service, including the
Model A wind estimator.

---

## Vercel: one link, whole app (recommended for sharing)

```bash
npm i -g vercel
vercel login
vercel --prod
```

That is the whole deploy. `vercel.json` already describes it: Vite builds the frontend to
`dist/`, and everything under `api/` becomes a serverless function.

| Route | Runtime | What it does |
|---|---|---|
| `/api/active-cyclones` | Node | GDACS live storm list |
| `/api/cyclone-track` | Node | One storm's observed track and GDACS cone |
| `/api/predict/trajectory` | **Python** | The trained forecast |
| `/api/predict/wind` | Node | Returns 503 with an explanation (see below) |

Vercel installs the function's Python dependencies from the **project root**
`requirements.txt`, not from a file beside the entrypoint. That is why the root file is
the minimal serving set (numpy) and the full stack lives in `requirements-ml.txt`: with
torch and its CUDA dependencies, pip resolves to 5.8 GB against a 500 MB function limit.

### Why the forecast runs without PyTorch

PyTorch is **534 MB** installed. A Vercel serverless function is capped at **250 MB**
unzipped, so torch cannot be deployed there at any setting.

The trained model is small (149,344 parameters) and entirely standard — an MLP, a
LayerNorm, one GRU layer, two linear heads — so `ml/numpy_forecaster.py` runs its forward
pass in numpy alone (34 MB). `ml/export_numpy_weights` converts the torch checkpoint into
`models/bob_trajectory_gru.npz` (544 KB), which is committed, and
`app/trajectory.py` prefers it over the `.pt` wherever both exist.

`tests/test_numpy_forecaster.py` pins the numpy path against PyTorch on random, real and
extreme inputs **including the checkpoint actually being served**, and asserts that
importing the serving path pulls in neither torch nor pandas. That test is the reason a
hand-written forward pass is trustworthy here: serving is where the model meets users, so
a divergence would be a wrong forecast rather than a failing test.

To regenerate the exported weights after retraining:

```bash
python -m ml.numpy_forecaster --checkpoint models/bob_trajectory_gru.pt
```

### Local install

```bash
pip install -r requirements.txt       # serving only: numpy
pip install -r requirements-ml.txt    # training and the full FastAPI service
pip install -r requirements-dev.txt   # the above plus pytest and linters
```

### What does not work on Vercel

The **wind estimator** (Model A) is a ResNet-18 and needs torch, so it cannot run there.
`api/predict/wind.ts` returns 503 with that explanation rather than letting the UI report
a vague failure. Its checkpoint is not in the repository either. Everything else — the
map, NASA GIBS layers, the GDACS feed and tracks, the Open-Meteo panel, and the full
trajectory forecast with its cone — works.

### Pointing the frontend at an external API instead

Leave `VITE_API_URL` unset and the browser calls the same-origin `/api/*` routes above.
Set it at build time and the browser calls `<VITE_API_URL>/predict/trajectory` directly,
which is how you would front a Cloud Run FastAPI service with a Vercel frontend.

---

# 🚀 Cloud Run: full PyTorch service

This guide explains how to deploy CycloneSense AI to **Google Cloud Run** with public access, resolving the *"You don't have access to this page"* error when accessing from mobile devices or other browsers.

---

## 🛑 Why "You don't have access to this page" Happens

When you run an app in the AI Studio development environment, the URL looks like:
`https://ais-dev-<hash>-<project-number>.<region>.run.app`

- **`ais-dev-...`** is a **private sandboxed development URL** protected by Google Cloud Identity-Aware Proxy (IAP) / Cloud Run IAM.
- Opening this URL on your phone or sending it to another person results in:
  > **"You don't have access to this page"** / **HTTP 403 Forbidden**
  because their browser is not signed in with the Google developer account that created the workspace.

---

## ✅ Solution 1: Deploy Directly to Google Cloud Run (Recommended for Production)

Deploying CycloneSense AI directly to your own Google Cloud Run service makes it publicly accessible with your own permanent HTTPS URL.

### Option A: 1-Line Command (Direct from Source)

Make sure you have [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and initialized:

```bash
# 1. Login and set your Google Cloud Project
gcloud auth login
gcloud config set project <YOUR_GCP_PROJECT_ID>

# 2. Deploy directly from source to Cloud Run with public unauthenticated access:
gcloud run deploy cyclonesense-ai \
  --source . \
  --region us-central1 \
  --platform managed \
  --port 3000 \
  --memory 1Gi \
  --cpu 1 \
  --allow-unauthenticated
```

> 🔑 **Key Flag:** `--allow-unauthenticated` is what makes the website open on **any mobile phone, tablet, or browser without asking for login**!

### Option B: Using the Included Shell / PowerShell Scripts

The project includes pre-configured deployment scripts:

**On Windows (PowerShell):**
```powershell
# If you don't have gcloud installed yet, install it via:
winget install Google.CloudSDK

# Then run:
.\deploy-cloudrun.ps1
```

**On Linux / macOS:**
```bash
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh
```

---

## 🛠️ Fixing: `gcloud : The term 'gcloud' is not recognized` on Windows

If you get this error in PowerShell:
```
gcloud : The term 'gcloud' is not recognized as the name of a cmdlet, function, script file...
```

It means the Google Cloud SDK is not yet installed on your Windows PC. Choose either method:

### Method 1: Install with 1 command in PowerShell (Easiest)
Run this in your PowerShell window:
```powershell
winget install Google.CloudSDK
```
*(When prompted, accept the license agreements. Once finished, close PowerShell and open a new PowerShell window).*

### Method 2: Official Windows Installer
1. Download the installer: https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
2. Run the installer and keep the default checkboxes selected (including "Start Google Cloud SDK Shell").
3. Once completed, restart PowerShell and you can run `gcloud` anywhere.

### Method 3: No Local Install Needed (Cloud Shell in Browser)
If you do not want to install software on your PC:
1. Open https://shell.cloud.google.com in your web browser.
2. Clone or upload your repository folder.
3. Run: `gcloud run deploy cyclonesense-ai --source . --port 3000 --allow-unauthenticated`
Cloud Shell already has `gcloud`, Docker, and Python pre-installed.

### Option C: Using Google Cloud Build (`cloudbuild.yaml`)

Run Cloud Build to containerize and deploy:
```bash
gcloud builds submit --config cloudbuild.yaml
```

---

## ✅ Solution 2: Instant Public Link via AI Studio "Share" Button

If you want to view this app right now on your phone without installing the `gcloud` CLI:

1. Look at the top navigation bar of **Google AI Studio**.
2. Click the **"Share"** button in the upper-right corner.
3. Copy the generated preview link (starts with `https://ais-pre-...`).
4. Paste that link into Safari, Chrome, or Firefox on your mobile phone.
5. The `ais-pre-...` link is publicly shared and does **not** block non-authenticated devices.

---

## 📱 Solution 3: Local Network (Wi-Fi) Access

To run locally on your laptop and access from a phone on the same Wi-Fi:

1. Start the server on your computer:
   ```bash
   npm run dev
   ```
2. Find your computer's local IP address:
   - **Windows:** Run `ipconfig` (look for IPv4 address, e.g., `192.168.1.50`)
   - **macOS / Linux:** Run `ifconfig` or `ip a` (look for `inet`, e.g., `192.168.1.50`)
3. On your phone's browser, navigate to:
   ```
   http://192.168.1.50:3000
   ```

---

## 🐳 Docker Deployment Details

The root `Dockerfile` is built with a lightweight Node 22 Alpine multi-stage build:
- **Build stage:** `npm ci` + `npm run build` (builds both Vite frontend and bundles Express server into `dist/server.cjs`)
- **Runtime stage:** Node 22 Alpine, copies `dist/`, exposes port 3000, and runs `node dist/server.cjs`.

To test the container locally:
```bash
docker build -t cyclonesense-ai .
docker run -p 3000:3000 cyclonesense-ai
```
Open `http://localhost:3000`.
