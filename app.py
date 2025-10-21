# app.py — TFLite-only FastAPI server

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import numpy as np
import psutil
import io, os, time

# =======================
# Config (env-overridable)
# =======================
TFLITE_PATH = os.getenv("TFLITE_PATH", "model_dynamic.tflite")
TFLITE_THREADS = int(os.getenv("TFLITE_NUM_THREADS", "2"))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.txt")
IMG_W = int(os.getenv("IMG_W", "224"))
IMG_H = int(os.getenv("IMG_H", "224"))
TOPK = int(os.getenv("TOPK", "5"))

# ===========
# TFLite init
# ===========
import tensorflow as tf  # using TF's bundled TFLite interpreter
assert os.path.exists(TFLITE_PATH), f"TFLite file not found: {TFLITE_PATH}"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=TFLITE_THREADS)
interpreter.allocate_tensors()
_in = interpreter.get_input_details()[0]
_out = interpreter.get_output_details()[0]

IN_DTYPE = _in["dtype"]
IN_SCALE, IN_ZERO = (_in.get("quantization") or (0.0, 0))
OUT_DTYPE = _out["dtype"]
OUT_SCALE, OUT_ZERO = (_out.get("quantization") or (0.0, 0))

print(f"[engine] tflite • {TFLITE_PATH} • in={IN_DTYPE} q=({IN_SCALE},{IN_ZERO}) • out={OUT_DTYPE} q=({OUT_SCALE},{OUT_ZERO})")

# ==============
# Helper loading
# ==============
def load_class_names(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

class_names = load_class_names(CLASS_NAMES_PATH)

# =========================
# Preprocess / Inference
# =========================
def preprocess_pil(pil_im: Image.Image) -> np.ndarray:
    """Return NumPy (1,H,W,3) float32 in [0,1]. Adjust if your training used another normalization."""
    im = pil_im.convert("RGB").resize((IMG_W, IMG_H))
    x = np.asarray(im, np.float32) / 255.0
    return x[None, ...]  # (1,H,W,3)

def _to_input_dtype(x: np.ndarray) -> np.ndarray:
    """Match model's expected input dtype (float32 or int8)."""
    if IN_DTYPE == np.float32:
        return x.astype(np.float32, copy=False)
    if IN_DTYPE == np.int8:
        # If your int8 model expects a specific pre-normalization, adjust here prior to quantization.
        if IN_SCALE and IN_SCALE > 0:
            q = np.round(x / IN_SCALE + IN_ZERO).astype(np.int8)
            return np.clip(q, -128, 127)
        return x.astype(np.int8)
    return x.astype(IN_DTYPE)

def _from_output_dtype(y: np.ndarray) -> np.ndarray:
    """Dequantize logits if needed to float32 for softmax."""
    if OUT_DTYPE == np.int8 and OUT_SCALE and OUT_SCALE > 0:
        return OUT_SCALE * (y.astype(np.float32) - OUT_ZERO)
    return y.astype(np.float32, copy=False)

def infer(x_np: np.ndarray) -> np.ndarray:
    """Run interpreter and return float32 logits."""
    x_in = _to_input_dtype(x_np)
    interpreter.set_tensor(_in["index"], x_in)
    interpreter.invoke()
    y = interpreter.get_tensor(_out["index"])
    return _from_output_dtype(y)

def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=axis, keepdims=True)

# =========================
# FastAPI app + middleware
# =========================
app = FastAPI(title="TFLite Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process = psutil.Process(os.getpid())
rolling = {"count": 0, "sum_ms": 0.0, "max_ms": 0.0}

@app.middleware("http")
async def timing_mw(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    rolling["count"] += 1
    rolling["sum_ms"] += dt_ms
    if dt_ms > rolling["max_ms"]:
        rolling["max_ms"] = dt_ms
    resp.headers["X-Request-Time-ms"] = f"{dt_ms:.2f}"
    return resp

@app.get("/health")
def health():
    return {"status": "ok", "engine": "tflite", "threads": TFLITE_THREADS}

@app.get("/perf")
def perf():
    rss_mb = process.memory_info().rss / (1024**2)
    return {
        "requests": rolling["count"],
        "avg_ms": (rolling["sum_ms"] / rolling["count"]) if rolling["count"] else 0.0,
        "max_ms": rolling["max_ms"],
        "rss_mb": round(rss_mb, 1),
        "threads": process.num_threads(),
        "cpu_percent": process.cpu_percent(interval=0.1),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    rss_before = process.memory_info().rss

    x = preprocess_pil(img)
    t0 = time.perf_counter()
    logits = infer(x)                      # (1, C)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    probs = softmax_np(logits, axis=-1)[0]
    # Top-K
    k = min(TOPK, probs.shape[-1])
    top_idx = np.argpartition(probs, -k)[-k:]
    top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]

    results = []
    for i in top_idx:
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        results.append({"index": int(i), "class": label, "score": float(probs[i])})

    rss_after = process.memory_info().rss
    rss_mb = rss_after / (1024**2)

    return JSONResponse({
        "topk": results,
        "timing_ms": {"inference": round(infer_ms, 2)},
        "memory": {"rss_mb": round(rss_mb, 1), "delta_mb": round((rss_after - rss_before)/(1024**2), 3)},
        "threads": process.num_threads()
    })
