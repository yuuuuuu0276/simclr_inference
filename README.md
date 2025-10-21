# SimCLR Inference Backend (FastAPI)

Backend service for image classification using a SimCLR encoder with a linear head.  
Supports **quantized (TFLite)** inference and (in `app_v1.py`) a fallback to a **naive TF** path.

## Contents

- **`app.py`** — FastAPI app for **quantized inference** (primary).
- **`app_v1.py`** — Older FastAPI app supporting **both** naive TF and quantized TFLite inference.
- **`export_tflite.py`** — Script to **export / quantize** a TensorFlow model to **TFLite**.
- **`class_names.txt`** — Line-separated class labels.
- **`saved_model/`** — TensorFlow SavedModel directory This folder is **not** in this repo. You can access it here: "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/" and put it in folder saved_model so that is ha `saved_model/saved_model.pb`, `saved_model/variables/variables.index`, `saved_model/variables/variables.data-00000-of-00001`
- **`best_linear_probe3.weights.h5`** — Linear head weights.
- **`model_dynamic.tflite`** — Quantized model produced by `export_tflite.py`.
- **`Dockerfile.backend` / `docker-compose.yaml`** — Containerization (FastAPI + optional Nginx frontend).

---

## Running (Docker Compose, recommended)

```bash
# Build & start backend (and frontend if present)
docker compose up -d --build

# Check health from the instance or your machine (depending on exposure):
curl http://localhost:8000/health
