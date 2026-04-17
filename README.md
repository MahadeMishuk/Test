# AI Eyes on the Road — User Manual

**Deployment platform: RunPod GPU only.**  
All execution happens inside a Docker container on a RunPod cloud GPU instance (A40 or equivalent).  
Code is edited locally, synced via rsync, and run exclusively via `docker-compose.gpu.yml`.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [One-Time Setup](#3-one-time-setup)
4. [Deploy to RunPod](#4-deploy-to-runpod)
5. [Access the Dashboard](#5-access-the-dashboard)
6. [Managing the Container](#6-managing-the-container)
7. [Re-Deploy After Code Changes](#7-re-deploy-after-code-changes)
8. [Using the Dashboard](#8-using-the-dashboard)
9. [REST API Reference](#9-rest-api-reference)
10. [Configuration Reference](#10-configuration-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

### Local machine (Mac/Linux — for editing and syncing only)

| Tool | Install |
|---|---|
| Git | pre-installed on Mac, or `brew install git` Install Homebrew first: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" | 
| rsync | `brew install rsync` (Mac) |
| SSH key | `~/.ssh/id_ed25519` — must be registered in RunPod |

### RunPod

| Requirement | Notes |
|---|---|
| RunPod account | [runpod.io](https://www.runpod.io) |
| GPU pod running | A40 (48 GB VRAM) recommended |
| SSH key registered | RunPod dashboard → Settings → SSH Keys |
| Pod exposes port 5001 | Set when creating the pod |

---

## 2. Project Structure

```
Pothole-I/
├── app.py                     # Flask + SocketIO entry point
├── config.py                  # All settings (env-var driven)
├── requirements.txt           # Python dependencies
│
├── pipeline/
│   ├── processor.py           # Per-frame CV orchestrator
│   ├── segmentation.py        # DeepLabV3 road mask (runs on CUDA)
│   ├── lane_detection.py      # Canny + Hough lane polygon
│   ├── detection.py           # YOLO pothole + COCO objects (CUDA)
│   └── depth.py               # Monocular distance estimation
│
├── models/
│   └── model_manager.py       # Lazy model loader → cuda:0
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
├── uploads/                   # Uploaded video files (volume-mounted)
├── outputs/                   # Annotated output videos (volume-mounted)
├── .cache/torch/              # Pretrained weight cache (volume-mounted)
│
├── Dockerfile.gpu             # CUDA 12.4 + cuDNN image (RunPod)
├── docker-compose.gpu.yml     # GPU compose — the only compose file used
├── .env.gpu                   # RunPod secrets — NEVER commit
└── scripts/
    ├── deploy_runpod.sh       # One-command sync + build + start
    └── verify_gpu.py          # GPU readiness checker
```

---

## 3. One-Time Setup

### 3.1 — Register your SSH key in RunPod

1. Open RunPod dashboard → **Settings** → **SSH Keys**
2. Paste the contents of `~/.ssh/id_ed25519.pub`
3. Save

Verify your local key exists:
```bash
cat ~/.ssh/id_ed25519.pub
```

### 3.2 — Create a RunPod pod

1. RunPod dashboard → **Deploy** → **GPU Pod**
2. Select GPU: **A40** (48 GB VRAM) or equivalent
3. Template: **RunPod PyTorch 2.4.1** (ships with CUDA 12.4, Docker, nvidia-container-toolkit)
4. Under **Expose Ports**, add port **`5001`** (Flask app) — port 22 is included by default
5. Click **Deploy**
6. Note your pod's **SSH host and port** from the Connect panel

### 3.3 — Configure .env.gpu

Edit `.env.gpu` in the project root on your local machine:

```env
# Flask
SECRET_KEY=<run: python3 -c "import secrets; print(secrets.token_hex(32))">
FLASK_ENV=production
DEBUG=False
HOST=0.0.0.0
PORT=5001

# GPU inference — mandatory
INFERENCE_DEVICE=cuda:0

# Trained pothole model (place best.pt at ./models/best.pt before deploying)
MODEL_PATH=/app/models/best.pt
POTHOLE_MODEL_PATH=/app/models/best.pt

# ML models
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

# GPU performance — process every frame
FRAME_SKIP=1
MAX_FRAME_WIDTH=640
MAX_UPLOAD_MB=500

# Prevent CUDA memory fragmentation on long sessions
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Generate SECRET_KEY:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

**Gmail App Password** (for email reports):
1. Google Account → Security → 2-Step Verification → App passwords
2. Create for "Mail" → paste the 16-character code as `SMTP_PASSWORD`

### 3.4 — Update deploy_runpod.sh with your pod's IP and port

Open [scripts/deploy_runpod.sh](scripts/deploy_runpod.sh) and set:

```bash
RUNPOD_HOST="<your pod IP>"      # e.g. 213.173.188.103
RUNPOD_PORT="<your pod SSH port>" # e.g. 36713
```

---

## 4. Deploy to RunPod

All deployment uses a single script. Run this from your local machine's terminal.

```bash
cd ~/Desktop/Pothole-I
bash scripts/deploy_runpod.sh
```

### What the script does

| Step | Action |
|---|---|
| **1 — Sync** | rsync project files to `/workspace/Pothole-I` on RunPod (skips `.git`, `.cache`, build artifacts, large uploads) |
| **2 — Build** | Builds `Dockerfile.gpu` on RunPod — CUDA 12.4, PyTorch 2.4.1+cu124, all dependencies |
| **3 — Start** | Runs `docker compose -f docker-compose.gpu.yml up -d` on RunPod |

> **First build time: 5–15 minutes** — PyTorch CUDA wheels are ~2.5 GB.  
> Subsequent deploys (code changes only): ~1–2 minutes (layer cache hit).

### What gets volume-mounted (persists across container restarts)

| Host path (RunPod) | Container path | Contents |
|---|---|---|
| `./uploads` | `/app/uploads` | Uploaded video files |
| `./outputs` | `/app/outputs` | Annotated output videos |
| `./static/snapshots` | `/app/static/snapshots` | Pothole frame crops |
| `./database` | `/app/database` | SQLite database |
| `./models` | `/app/models` | YOLO weights + custom pothole model |
| `./.cache/torch` | `/root/.cache/torch` | DeepLabV3 weight cache (~170 MB, saves re-download) |

---

## 5. Access the Dashboard

### Option A — SSH tunnel (always works, no extra RunPod config)

Open a new terminal on your local machine:

```bash
ssh -L 5001:localhost:5001 -i ~/.ssh/id_ed25519 -p <RUNPOD_PORT> root@<RUNPOD_HOST>
```

Keep that terminal open, then open: **http://localhost:5001**

### Option B — Direct RunPod proxy URL

1. Pod card → **Connect** → **HTTP Services**
2. If port 5001 is listed, click the link directly
3. If not listed, click **+ Expose Port** → add `5001` → Save

---

## 6. Managing the Container

All management commands are run via SSH from your local terminal.

### SSH into the pod

```bash
ssh -i ~/.ssh/id_ed25519 -p <RUNPOD_PORT> root@<RUNPOD_HOST>
cd /workspace/Pothole-I
```

### Container lifecycle

```bash
# Check status and health
docker compose -f docker-compose.gpu.yml ps

# Tail live logs
docker compose -f docker-compose.gpu.yml logs -f

# Stop
docker compose -f docker-compose.gpu.yml down

# Start (image already built)
docker compose -f docker-compose.gpu.yml up -d

# Rebuild after Dockerfile or requirements.txt changes
docker compose -f docker-compose.gpu.yml up --build -d
```

### Open a shell inside the running container

```bash
docker exec -it ai-eyes-on-the-road-gpu bash
```

### Verify GPU is active inside the container

```bash
docker exec ai-eyes-on-the-road-gpu python3 -c \
  "import torch; print('CUDA:', torch.cuda.is_available()); \
   print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
GPU: NVIDIA A40
```

### Run the full GPU readiness check

```bash
docker exec ai-eyes-on-the-road-gpu python scripts/verify_gpu.py
```

Checks: CUDA availability, VRAM, PyTorch version, YOLO smoke test, nvidia-smi snapshot.

### Check GPU utilization in real time

```bash
# On the pod (outside the container)
watch -n 1 nvidia-smi
```

---

## 7. Re-Deploy After Code Changes

Edit code locally, then re-run the deploy script:

```bash
# On your local machine
cd ~/Desktop/Pothole-I
bash scripts/deploy_runpod.sh
```

rsync sends only changed files. The Docker layer cache means only affected layers rebuild.

- **Python file change only** → sync takes ~5 s, container restarts in ~10 s
- **requirements.txt change** → pip layer rebuilds (~3–5 min)
- **Dockerfile.gpu change** → full rebuild (~10–15 min)

---

## 8. Using the Dashboard

Open **http://localhost:5001** (SSH tunnel) or the RunPod proxy URL.

### 8.1 — Upload a Video for Processing

1. Click **Upload Video** in the top bar
2. Select a `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm` file (max 500 MB)
3. Click **Process**
4. The live annotated feed appears in the center panel in real time
5. The processed output video is saved to `outputs/` on the RunPod pod

### 8.2 — Live Camera Mode

1. Click **Start Camera**
2. Allow browser camera access when prompted
3. The browser captures frames via `getUserMedia` on your local device and streams them over WebSocket to the RunPod container
4. Fully functional — no GPU-side camera device access needed

### 8.3 — Detection Overlay Guide

| Color | Meaning |
|---|---|
| Red box | Pothole — DANGER (< 5 m) |
| Orange box | Pothole — CAUTION (5–15 m) |
| Green box | Pothole — safe distance (> 15 m) |
| Cyan box | General objects: cars, trucks, pedestrians |
| Green road overlay | Detected road surface (DeepLabV3 segmentation mask) |
| Cyan lane lines | Canny + Hough lane polygon overlay |

### 8.4 — Map View

- Click the **Map** tab to see all detected potholes on a Mapbox map
- Star markers indicate pothole GPS locations
- Click any marker for details: coordinates, confidence score, timestamp, frame snapshot

### 8.5 — Pothole Database Table

- **Potholes** tab lists all detections stored in SQLite
- Sort by confidence, date, or distance
- **Report** — sends an MDOT email report for that pothole
- **Report All Unreported** — batch-sends all pending reports
- **Delete** — removes the record from the database

### 8.6 — MDOT Email Reporting

The system sends structured pothole reports via email automatically.

1. `SMTP_USERNAME`, `SMTP_PASSWORD`, `REPORT_FROM_EMAIL` must be set in `.env.gpu`
2. Each report includes: GPS coordinates + Google Maps link, timestamp, confidence score, risk level, and a pothole frame snapshot image attached
3. Follow-ups are sent automatically after 7 days if no repair acknowledgment is received

---

## 9. REST API Reference

Base URL: `http://localhost:5001` (or RunPod proxy URL)

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
| `GET` | `/api/performance` | CPU / memory / GPU metrics |

**Examples:**
```bash
curl http://localhost:5001/api/health
curl http://localhost:5001/api/stats
curl http://localhost:5001/api/potholes
```

---

## 10. Configuration Reference

All variables are set in `.env.gpu`. Changes require re-deploying.

### ML Models

| Variable | Value (GPU) | Description |
|---|---|---|
| `INFERENCE_DEVICE` | `cuda:0` | GPU device — never change this |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO variant: n/s/m/l/x |
| `MODEL_PATH` | `/app/models/best.pt` | Trained pothole model |
| `POTHOLE_MODEL_PATH` | `/app/models/best.pt` | Legacy alias for MODEL_PATH |
| `SEGMENTATION_BACKEND` | `deeplabv3` | `deeplabv3` or `segformer` |
| `USE_SEGMENTATION` | `True` | Enable DeepLabV3 road mask |
| `USE_DEPTH_MIDAS` | `False` | Enable MiDaS depth (enable on GPU — fast) |

### Detection Thresholds

| Variable | Default | Description |
|---|---|---|
| `DETECTION_CONFIDENCE` | `0.35` | YOLO general confidence |
| `POTHOLE_CONFIDENCE` | `0.30` | Pothole-specific confidence |
| `USE_LANE_DETECTION` | `True` | Enable Canny + Hough lane mask |
| `LANE_MASK_COMBINE` | `intersect` | `intersect` or `union` for mask merging |

### Performance

| Variable | GPU Value | Description |
|---|---|---|
| `FRAME_SKIP` | `1` | Process every frame (GPU can handle it) |
| `MAX_FRAME_WIDTH` | `640` | Inference width in pixels |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Prevents VRAM fragmentation |

### Distance / Risk Zones

| Variable | Default | Description |
|---|---|---|
| `NEAR_DISTANCE` | `5.0` | Metres — danger zone (red alert) |
| `MEDIUM_DISTANCE` | `15.0` | Metres — caution zone (orange alert) |
| `ALERT_COOLDOWN_SEC` | `3.0` | Minimum seconds between alerts |

### Using a Custom Pothole Model

1. Place your `best.pt` at `models/best.pt` in the project root (local machine)
2. Re-run deploy — rsync will upload it; the volume mount makes it available in the container
3. Verify it loaded:
   ```bash
   docker exec ai-eyes-on-the-road-gpu python3 -c \
     "from models.model_manager import ModelManager
   from config import config
   mm = ModelManager(config); mm.initialize()
   print('Pothole model:', mm.pothole_yolo)"
   ```

Detection uses a 3-layer strategy:
1. Custom pothole model (if `best.pt` exists)
2. YOLOv8 COCO general model
3. OpenCV anomaly detection on the road mask

---

## 11. Troubleshooting

### Deploy script fails — SSH timeout

- Confirm the pod is running in RunPod dashboard (green status)
- Confirm `RUNPOD_HOST` and `RUNPOD_PORT` in `deploy_runpod.sh` match the current pod's Connect details (pods get new IPs on restart)

### Container won't start — check logs

```bash
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> \
  'cd /workspace/Pothole-I && docker compose -f docker-compose.gpu.yml logs --tail=100'
```

### CUDA not available inside container

```bash
# On the pod — check NVIDIA runtime
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi
```

If docker can't see the GPU:
```bash
# Should already be done on RunPod, but verify:
apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

### Out of VRAM

Check current VRAM usage:
```bash
docker exec ai-eyes-on-the-road-gpu nvidia-smi
```

Reduce memory pressure in `.env.gpu`:
```env
MAX_FRAME_WIDTH=416        # smaller inference resolution
FRAME_SKIP=2               # process every other frame
USE_DEPTH_MIDAS=False      # disable MiDaS if enabled
```

### DeepLabV3 weights re-downloading every restart

The `.cache/torch` volume should prevent this. If it's re-downloading:
```bash
# Verify the cache volume exists on the pod
ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST> \
  'ls /workspace/Pothole-I/.cache/torch/hub/checkpoints/'
```

If missing, the first run will download (~170 MB) and cache it for all future restarts.

### Video upload fails

- Check `MAX_UPLOAD_MB=500` in `.env.gpu`
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`
- Confirm the `uploads/` volume directory exists: `ls /workspace/Pothole-I/uploads/`

### Email reports not sending

Test SMTP connectivity inside the container:
```bash
docker exec ai-eyes-on-the-road-gpu python3 -c "
import smtplib, os
s = smtplib.SMTP('smtp.gmail.com', 587)
s.ehlo(); s.starttls()
s.login(os.environ['SMTP_USERNAME'], os.environ['SMTP_PASSWORD'])
print('SMTP OK')
s.quit()
"
```

### Pod IP changed after restart

RunPod assigns a new IP each time a pod restarts. Update `deploy_runpod.sh`:
```bash
RUNPOD_HOST="<new IP>"
RUNPOD_PORT="<new port>"
```

Then re-run `bash scripts/deploy_runpod.sh`.

---

## Quick Reference Card

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AI Eyes on the Road — RunPod GPU Quick Reference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPLOY (from local machine):
  bash scripts/deploy_runpod.sh

SSH INTO POD:
  ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST>

ACCESS APP (SSH tunnel):
  ssh -L 5001:localhost:5001 -i ~/.ssh/id_ed25519 -p <PORT> root@<HOST>
  → http://localhost:5001

TAIL LOGS:
  ssh ... 'cd /workspace/Pothole-I && \
    docker compose -f docker-compose.gpu.yml logs -f'

CONTAINER SHELL:
  docker exec -it ai-eyes-on-the-road-gpu bash

GPU CHECK:
  docker exec ai-eyes-on-the-road-gpu python scripts/verify_gpu.py

RESTART CONTAINER:
  docker compose -f docker-compose.gpu.yml down
  docker compose -f docker-compose.gpu.yml up -d
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

*AI Eyes on the Road · RunPod GPU Edition · YOLOv8 · DeepLabV3 · Flask · SocketIO*
