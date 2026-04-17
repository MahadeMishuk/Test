# AI Eyes on the Road — User Manual

Complete step-by-step guide for running the pothole detection system.  
Three deployment paths are covered: local Python, Docker (CPU), and RunPod GPU.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure Overview](#2-project-structure-overview)
3. [Environment Setup (.env)](#3-environment-setup-env)
4. [Path A — Run Locally (Python)](#4-path-a--run-locally-python)
5. [Path B — Run with Docker (CPU, any machine)](#5-path-b--run-with-docker-cpu-any-machine)
6. [Path C — Run on RunPod GPU](#6-path-c--run-on-runpod-gpu)
7. [Using the Dashboard](#7-using-the-dashboard)
8. [REST API Reference](#8-rest-api-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### For Local / Docker (CPU)

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | `python --version` |
| pip | latest | `pip install --upgrade pip` |
| Docker Desktop | 4.x | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) |
| Docker Compose | V2 (bundled) | `docker compose version` |
| Git | any | to clone the repo |

### For RunPod GPU

| Requirement | Notes |
|---|---|
| RunPod account | [runpod.io](https://www.runpod.io) |
| SSH key registered in RunPod | Settings → SSH Keys |
| rsync (local Mac/Linux) | `brew install rsync` on Mac |
| Pod with CUDA 12.x GPU | A40 recommended (48 GB VRAM) |

---

## 2. Project Structure Overview

```
Pothole-I/
├── app.py                     # Flask + SocketIO entry point
├── config.py                  # All env-var settings in one place
├── requirements.txt           # Python dependencies
│
├── pipeline/
│   ├── processor.py           # Per-frame CV orchestrator
│   ├── segmentation.py        # DeepLabV3 road mask
│   ├── lane_detection.py      # Canny + Hough lane polygon
│   ├── detection.py           # YOLO pothole + COCO objects
│   └── depth.py               # Monocular distance estimation
│
├── models/
│   └── model_manager.py       # Lazy model loader
│
├── tracking/                  # SORT tracker (Kalman + Hungarian)
├── alerts/                    # Audio + visual alert manager
├── reporting/                 # MDOT email reporting
├── database/                  # SQLAlchemy + SQLite
├── api/routes.py              # REST API blueprint
│
├── static/                    # CSS, JS, icons
├── templates/index.html       # Main dashboard HTML
│
├── uploads/                   # Uploaded video files (runtime)
├── outputs/                   # Annotated output videos (runtime)
├── .cache/torch/              # Pretrained model weight cache
│
├── Dockerfile                 # CPU image (Python 3.12 slim)
├── Dockerfile.gpu             # GPU image (CUDA 12.4 + cuDNN)
├── docker-compose.yml         # CPU compose
├── docker-compose.gpu.yml     # GPU compose (RunPod)
├── .env                       # CPU secrets (never commit)
├── .env.gpu                   # GPU secrets (never commit)
└── scripts/
    ├── deploy_runpod.sh       # One-command RunPod deploy
    └── verify_gpu.py          # GPU readiness checker
```

---

## 3. Environment Setup (.env)

The app reads all secrets from environment files. You must configure these before running anything.

### 3.1 CPU / Local

Edit `.env` in the project root:

```env
# Flask
SECRET_KEY=change-me-to-a-random-string-32-chars
DEBUG=False
HOST=0.0.0.0
PORT=5001

# ML Models
YOLO_MODEL=yolov8n.pt
SEGMENTATION_BACKEND=deeplabv3
USE_SEGMENTATION=True
USE_DEPTH_MIDAS=False

# Detection thresholds
DETECTION_CONFIDENCE=0.35
POTHOLE_CONFIDENCE=0.30

# Mapbox (required for map view)
MAPBOX_ACCESS_TOKEN=pk.your_mapbox_token_here
MAPBOX_STYLE=mapbox://styles/mapbox/dark-v11
DEFAULT_LAT=39.2904
DEFAULT_LNG=-76.6122
DEFAULT_ZOOM=13

# Email reporting (optional)
REPORT_FROM_EMAIL=your@gmail.com
SMTP_USERNAME=your@gmail.com
SMTP_PASSWORD=your-gmail-app-password
PERSONAL_REPORT_EMAIL=your@gmail.com

# Performance
FRAME_SKIP=2
MAX_FRAME_WIDTH=640
MAX_UPLOAD_MB=500
```

**Generate a SECRET_KEY:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

**Gmail App Password (for email reports):**
1. Go to Google Account → Security → 2-Step Verification → App passwords
2. Create a password for "Mail"
3. Paste the 16-character code as `SMTP_PASSWORD`

### 3.2 GPU / RunPod

Edit `.env.gpu`. It has the same keys plus:

```env
# Force GPU inference
INFERENCE_DEVICE=cuda:0

# Point to your trained pothole model (optional)
MODEL_PATH=/app/models/best.pt
POTHOLE_MODEL_PATH=/app/models/best.pt

# GPU performance
FRAME_SKIP=1                   # GPU can handle every frame
MAX_UPLOAD_MB=500
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 4. Path A — Run Locally (Python)

Use this for development or quick testing without Docker.

### Step 1 — Clone and enter the project

```bash
git clone <your-repo-url> Pothole-I
cd Pothole-I
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Mac / Linux
# venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> First install takes 3–8 minutes. PyTorch alone is ~1 GB.

### Step 4 — Configure environment

```bash
cp .env .env.backup    # keep a backup
# Edit .env — set SECRET_KEY and any API keys you want
```

### Step 5 — Run the app

```bash
python app.py
```

You should see:
```
* Running on http://0.0.0.0:5001
* SocketIO server started
```

### Step 6 — Open the dashboard

Open your browser: **http://localhost:5001**

The first page load triggers model loading (DeepLabV3 + YOLO).  
Wait ~30–60 seconds for the pipeline to be ready.

---

## 5. Path B — Run with Docker (CPU, any machine)

Use this for a clean, reproducible CPU deployment without touching Python on your host.

### Step 1 — Install Docker Desktop

Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) and start it.

Verify:
```bash
docker --version
docker compose version
```

### Step 2 — Clone the project

```bash
git clone <your-repo-url> Pothole-I
cd Pothole-I
```

### Step 3 — Configure .env

Edit `.env` with your SECRET_KEY and API keys (see Section 3.1).

### Step 4 — Build and start

First run (downloads base image + installs ~1 GB Python packages):
```bash
docker compose up --build
```

Subsequent runs (image already built, starts in seconds):
```bash
docker compose up
```

Run in background:
```bash
docker compose up -d
```

### Step 5 — Open the dashboard

**http://localhost:5001**

### Useful Docker commands

```bash
# Tail live logs
docker compose logs -f

# Stop the container
docker compose down

# Rebuild after code changes
docker compose up --build

# Open a shell inside the running container
docker exec -it ai-eyes-on-the-road bash

# Check container health
docker compose ps
```

### What the volumes do

| Host path | Container path | Purpose |
|---|---|---|
| `./uploads` | `/app/uploads` | Uploaded videos survive restarts |
| `./outputs` | `/app/outputs` | Annotated videos survive restarts |
| `./static/snapshots` | `/app/static/snapshots` | Pothole frame crops |
| `./database` | `/app/database` | SQLite DB survives restarts |
| `./models` | `/app/models` | Custom YOLO model (hot-swap) |
| `./.cache/torch` | `/root/.cache/torch` | DeepLabV3 weights cache (saves ~170 MB re-download) |

---

## 6. Path C — Run on RunPod GPU

This deploys the project into a Docker container on a RunPod cloud GPU pod.  
The GPU accelerates YOLO, segmentation, and depth estimation significantly.

### 6.1 — Create a RunPod Pod

1. Log in at [runpod.io](https://www.runpod.io)
2. Click **Deploy** → **GPU Pod**
3. Select a GPU (A40 recommended — 48 GB VRAM)
4. Set the template to **RunPod PyTorch 2.4.1** (CUDA 12.4 pre-installed)
5. Under **Expose Ports**, add:
   - `5001` (the Flask app)
   - `22` is already included for SSH
6. Click **Deploy**

### 6.2 — Add your SSH key to RunPod

1. RunPod dashboard → **Settings** → **SSH Keys**
2. Paste your **public key** (`~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`)
3. Save

### 6.3 — Get your pod's SSH connection details

In the pod card, click **Connect**. You'll see something like:

```
SSH:  ssh root@213.173.188.103 -p 36713 -i ~/.ssh/id_ed25519
```

Write down the **host IP** and **port** — you'll need them.

### 6.4 — Configure .env.gpu on your local machine

Edit `.env.gpu` in the project root before uploading:

```env
SECRET_KEY=<generate with: python3 -c "import secrets; print(secrets.token_hex(32))">
SMTP_PASSWORD=<your Gmail app password>
# Everything else is already set correctly
```

### 6.5 — Run the deploy script

From your local machine's terminal:

```bash
cd ~/Desktop/Pothole-I
bash scripts/deploy_runpod.sh
```

**What the script does automatically:**

| Step | Action |
|---|---|
| 1 | rsync project files to `/workspace/Pothole-I` on RunPod (skips `.git`, `.cache`, build artifacts) |
| 2 | Builds the GPU Docker image on RunPod (CUDA 12.4 + PyTorch 2.4.1+cu124) |
| 3 | Starts the container via `docker-compose.gpu.yml` |
| 4 | Prints access instructions |

> **First build takes 5–15 minutes** — PyTorch CUDA wheels alone are ~2.5 GB.  
> Subsequent deploys (code changes only) take ~1–2 minutes.

### 6.6 — Verify GPU is working

After the container starts, run:

```bash
ssh -i ~/.ssh/id_ed25519 -p 36713 root@<YOUR_POD_IP> \
  "docker exec ai-eyes-on-the-road-gpu python3 -c \
  \"import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))\""
```

Expected output:
```
CUDA: True
GPU: NVIDIA A40
```

Or run the full readiness check:
```bash
ssh -i ~/.ssh/id_ed25519 -p 36713 root@<YOUR_POD_IP> \
  "docker exec ai-eyes-on-the-road-gpu python scripts/verify_gpu.py"
```

### 6.7 — Access the dashboard

**Option A — SSH tunnel (works immediately, no RunPod UI changes needed):**

```bash
ssh -L 5001:localhost:5001 -i ~/.ssh/id_ed25519 -p 36713 root@<YOUR_POD_IP>
```

Keep this terminal open, then open **http://localhost:5001** in your browser.

**Option B — RunPod proxy URL:**

1. In the pod card, click **Connect** → **HTTP Services**
2. If port 5001 appears, click the link to open it directly
3. If it doesn't appear, click **+ Expose Port** → add `5001` → Save

### 6.8 — Tail logs on RunPod

```bash
ssh -i ~/.ssh/id_ed25519 -p 36713 root@<YOUR_POD_IP> \
  'cd /workspace/Pothole-I && docker compose -f docker-compose.gpu.yml logs -f'
```

### 6.9 — Manage the container on RunPod

All commands run via SSH from your local terminal:

```bash
# SSH into the pod first
ssh -i ~/.ssh/id_ed25519 -p 36713 root@<YOUR_POD_IP>

# Then on the pod:
cd /workspace/Pothole-I

# Check status
docker compose -f docker-compose.gpu.yml ps

# Stop
docker compose -f docker-compose.gpu.yml down

# Restart
docker compose -f docker-compose.gpu.yml up -d

# Rebuild after code changes
docker compose -f docker-compose.gpu.yml up --build -d

# Open a shell inside the running container
docker exec -it ai-eyes-on-the-road-gpu bash
```

### 6.10 — Re-deploy after code changes

From your local machine, just re-run:
```bash
bash scripts/deploy_runpod.sh
```

The script rsyncs only changed files, then rebuilds and restarts.

---

## 7. Using the Dashboard

Open **http://localhost:5001** (or the RunPod URL).

### 7.1 — Upload a Video

1. Click **Upload Video** in the top bar
2. Select an `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm` file (max 500 MB)
3. Click **Process**
4. Watch the live annotated feed appear in the center panel
5. After processing, the annotated output is saved to `outputs/`

### 7.2 — Live Camera Mode

1. Click **Start Camera**
2. Allow browser camera access when prompted
3. The browser captures frames via `getUserMedia` and streams them over WebSocket
4. Works on both local and RunPod deployments (camera stays on your device)

### 7.3 — Reading the Detection Overlay

| Color | Meaning |
|---|---|
| Red box | Pothole — DANGER (< 5 m) |
| Yellow box | Pothole — CAUTION (5–15 m) |
| Green box | Pothole — safe distance (> 15 m) |
| Blue box | General objects (cars, trucks, pedestrians) |
| Green road overlay | Detected road surface (segmentation mask) |
| Cyan lines | Lane detection overlay |

### 7.4 — Map View

- Click the **Map** tab to see all detected potholes plotted on a Mapbox map
- Star markers indicate pothole locations
- Click a marker for details: GPS, confidence, timestamp, frame snapshot

### 7.5 — Pothole Database Table

- The **Potholes** tab lists all detected potholes from the database
- Sort by confidence, date, or distance
- Click **Report** to send an MDOT report email for that pothole
- Click **Delete** to remove a record

### 7.6 — MDOT Email Reporting

1. Configure `SMTP_USERNAME`, `SMTP_PASSWORD`, and `REPORT_FROM_EMAIL` in `.env`
2. In the Potholes tab, click **Report** on any pothole
3. Or click **Report All Unreported** to batch-send
4. Reports include: GPS coordinates, timestamp, confidence score, risk level, and a frame snapshot image
5. The system auto-sends follow-ups after 7 days if unacknowledged

---

## 8. REST API Reference

Base URL: `http://localhost:5001`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/api/potholes` | List all potholes (paginated) |
| `GET` | `/api/potholes/<id>` | Single pothole details |
| `DELETE` | `/api/potholes/<id>` | Delete a record |
| `POST` | `/api/potholes/<id>/report` | Submit MDOT report |
| `GET` | `/api/potholes/<id>/report-preview` | Preview report text |
| `POST` | `/api/potholes/<id>/mark-crossed` | Mark pothole as crossed |
| `GET` | `/api/stats` | Detection statistics |
| `GET` | `/api/alerts` | Recent alert history |
| `GET` | `/api/map-data` | GeoJSON for map rendering |
| `GET` | `/api/performance` | CPU/memory/GPU metrics |

**Example — get all potholes:**
```bash
curl http://localhost:5001/api/potholes
```

**Example — check health:**
```bash
curl http://localhost:5001/api/health
```

---

## 9. Configuration Reference

All variables are set in `.env` (CPU) or `.env.gpu` (GPU).

### ML Models

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | YOLO variant: n/s/m/l/x (n=fastest) |
| `POTHOLE_MODEL_PATH` | `models/pothole_yolov8.pt` | Custom pothole model path (optional) |
| `SEGMENTATION_BACKEND` | `deeplabv3` | `deeplabv3` \| `segformer` \| `geometric` |
| `USE_SEGMENTATION` | `True` | Enable/disable road segmentation |
| `USE_DEPTH_MIDAS` | `False` | Enable MiDaS neural depth (slow on CPU) |
| `INFERENCE_DEVICE` | `cpu` | `cpu` or `cuda:0` |

### Detection

| Variable | Default | Description |
|---|---|---|
| `DETECTION_CONFIDENCE` | `0.35` | YOLO general confidence threshold |
| `POTHOLE_CONFIDENCE` | `0.30` | Pothole-specific confidence |
| `NMS_THRESHOLD` | `0.45` | Non-max suppression IoU |
| `POTHOLE_MIN_AREA` | `800` | Min contour area (px²) for CV detection |
| `USE_LANE_DETECTION` | `True` | Enable Canny+Hough lane polygon |
| `LANE_MASK_COMBINE` | `intersect` | `intersect` or `union` for mask merging |

### Performance

| Variable | Default | Description |
|---|---|---|
| `FRAME_SKIP` | `2` | Process every Nth frame (1=every frame) |
| `MAX_FRAME_WIDTH` | `640` | Resize width before inference |
| `JPEG_QUALITY` | `75` | SocketIO stream JPEG quality |

### Distance / Risk

| Variable | Default | Description |
|---|---|---|
| `NEAR_DISTANCE` | `5.0` | Metres — danger zone (red alert) |
| `MEDIUM_DISTANCE` | `15.0` | Metres — caution zone (yellow alert) |
| `ALERT_COOLDOWN_SEC` | `3.0` | Minimum seconds between alerts |

### Custom Pothole Model

If you have a custom YOLOv8 model trained on pothole data:

1. Copy `best.pt` to `models/pothole_yolov8.pt`
2. Set in `.env`:
   ```env
   POTHOLE_MODEL_PATH=models/pothole_yolov8.pt
   ```
3. Restart the app or container

The pipeline uses a 3-layer fallback:
1. Custom pothole model (if file exists)
2. YOLO COCO general model
3. OpenCV anomaly detection on the road mask

---

## 10. Troubleshooting

### App won't start — port already in use

```bash
# Find what's using port 5001
lsof -i :5001
# Kill it
kill -9 <PID>
```

Or change the port in `.env`:
```env
PORT=5002
```

### Docker build fails — network error

```bash
# Retry with no cache
docker compose build --no-cache
```

### Container starts but page doesn't load

Check logs for errors:
```bash
docker compose logs -f
```

Wait up to 120 seconds — the health check has a 2-minute start window for model loading.

### CUDA not available on RunPod

```bash
# Check on the pod
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi
```

If `nvidia-smi` works but Docker can't see the GPU:
```bash
# On the pod (RunPod has this pre-installed, but check)
apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

### DeepLabV3 / model download fails inside container

The `.cache/torch` volume mounts the weight cache from the host.  
If the download failed mid-way, clear the cache:

```bash
# On your local machine (CPU Docker)
rm -rf .cache/torch

# Or on RunPod
ssh -i ~/.ssh/id_ed25519 -p 36713 root@<IP> "rm -rf /workspace/Pothole-I/.cache/torch"
```

Then restart the container — it will re-download cleanly.

### Video upload fails

- Check `MAX_UPLOAD_MB` in `.env` (default 500)
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`
- Check the `uploads/` directory is writable

### Email reports not sending

1. Confirm `SMTP_USERNAME` and `SMTP_PASSWORD` are set
2. Use a Gmail App Password (not your normal password) — see Section 3.1
3. Test manually:
   ```bash
   python3 -c "
   import smtplib
   s = smtplib.SMTP('smtp.gmail.com', 587)
   s.ehlo(); s.starttls()
   s.login('your@gmail.com', 'your-app-password')
   print('SMTP OK')
   s.quit()
   "
   ```

### Performance is slow on CPU

Try these settings in `.env`:
```env
USE_SEGMENTATION=False     # biggest speedup — disables DeepLabV3
FRAME_SKIP=3               # process every 3rd frame
MAX_FRAME_WIDTH=416        # smaller inference size
USE_LANE_DETECTION=False   # disable Canny+Hough pass
```

Expected FPS by config:

| Config | Approx FPS |
|---|---|
| Full pipeline (CPU) | 3–8 |
| No segmentation (CPU) | 10–15 |
| GPU (A40, full pipeline) | 25–60 |

---

*AI Eyes on the Road — Built with YOLOv8 · OpenCV · Flask · SocketIO · SQLAlchemy*
