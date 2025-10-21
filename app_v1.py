# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras import layers
from fastapi.middleware.cors import CORSMiddleware
import io, time, os


import psutil
process = psutil.Process(os.getpid())
rolling = {"count": 0, "sum_ms": 0.0, "max_ms": 0.0}

# at top
USE_TFLITE = os.getenv("USE_TFLITE", "0") == "1"
# TFLITE_PATH = os.getenv("TFLITE_PATH", "./model_dynamic.tflite")

TFLITE_PATH = './model_dynamic.tflite'

if USE_TFLITE:
    import tensorflow as tf  # <- use TF's built-in TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=int(os.getenv("TFLITE_NUM_THREADS", "2")))
    interpreter.allocate_tensors()
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    def infer(x_np: np.ndarray) -> np.ndarray:
        interpreter.set_tensor(in_det["index"], x_np.astype(in_det["dtype"], copy=False))
        interpreter.invoke()
        return interpreter.get_tensor(out_det["index"])
else:
    import tensorflow as tf
    # ... build/load your Keras model as before ...
    def infer(x_np: np.ndarray) -> np.ndarray:
        return model(tf.convert_to_tensor(x_np), training=False).numpy()

# inside /predict just call `logits = infer(x_np)` instead of model(...)



app = FastAPI(title="SimCLR Linear Probe Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or lock to your site later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def _timing_middleware(request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    # lightweight rolling stats (single worker)
    rolling["count"] += 1
    rolling["sum_ms"] += dt_ms
    if dt_ms > rolling["max_ms"]:
        rolling["max_ms"] = dt_ms
    # also expose per-response timing header (handy in browser)
    resp.headers["X-Request-Time-ms"] = f"{dt_ms:.2f}"
    return resp


CLASS_NAMES_PATH = "./class_names.txt"
SIMCLR_DIR = "./saved_model" 
# SIMCLR_DIR = "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/"  # <-- folder, not .pb
CKPT_PATH = "./best_linear_probe3.weights.h5"  # match your filename
NUM_CLASSES = 40
IMG_SIZE = (224, 224)   # change if your encoder expects a different size
TOPK = 5

def load_class_names(path: str) -> List[str]:
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names

class SimCLRLinearClassifier(tf.keras.Model):
    def __init__(self, simclr_path, num_classes, weight_decay=0.0):
        super().__init__()
        # Encoder is a SavedModel with a dict-like output (expects key 'final_avg_pool')
        self.encoder = tf.saved_model.load(simclr_path)
        self.head = layers.Dense(
            num_classes,
            name="linear_head",
            kernel_regularizer=(tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None)
        )

    @tf.function
    def call(self, images, training=False):
        feats = self.encoder(images, trainable=False)['final_avg_pool']
        if not training:
            feats = tf.stop_gradient(feats)
        logits = self.head(feats, training=training)
        return logits

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3] in [0,1]
    arr = np.expand_dims(arr, axis=0)                # [1,H,W,3]
    return arr

@app.on_event("startup")
def startup():
    global model, class_names
    class_names = load_class_names(CLASS_NAMES_PATH)
    if len(class_names) != NUM_CLASSES:
        # Not fatal, but warn
        print(f"[warn] class_names({len(class_names)}) != NUM_CLASSES({NUM_CLASSES})")

    model = SimCLRLinearClassifier(SIMCLR_DIR, NUM_CLASSES, weight_decay=0.0)
    # Build model & load weights
    dummy = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    model.build(IMG_SIZE)
    _ = model(dummy, training=False)
    model.load_weights(CKPT_PATH)
    _ = model(dummy, training=False)


    model.load_weights(CKPT_PATH)
    _ = model(dummy, training=False)
    print("[ready] model loaded & warmed")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/perf")
def perf():
    rss_mb = process.memory_info().rss / (1024**2)
    return {
        "requests": rolling["count"],
        "avg_ms": (rolling["sum_ms"] / rolling["count"]) if rolling["count"] else 0.0,
        "max_ms": rolling["max_ms"],
        "rss_mb": round(rss_mb, 1),
        "threads": process.num_threads(),
        # simple CPU usage sample (last 0.1s window)
        "cpu_percent": process.cpu_percent(interval=0.1),
    }


def preprocess_pil(pil_im, size=(224,224)):
    im = pil_im.convert("RGB").resize(size)
    x = np.asarray(im, np.float32) / 255.0
    # x = (x - 0.5) * 2.0      # [-1,1] if that’s what you trained with
    return x[None, ...]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # memory before
    rss_before = process.memory_info().rss

    # time just the forward pass
    x_np = preprocess_pil(img)      # NumPy array
    t0 = time.perf_counter()
    logits = infer(x_np)            
    # x = preprocess_pil(img)
    # t0 = time.perf_counter()
    # logits = model(tf.convert_to_tensor(x), training=False)


    infer_ms = (time.perf_counter() - t0) * 1000.0

    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    topk_idx = probs.argsort()[-TOPK:][::-1].tolist()
    results = [
        {"class": class_names[i] if i < len(class_names) else f"class_{i}",
         "score": float(probs[i]),
         "index": int(i)}
        for i in topk_idx
    ]

    # memory after
    rss_after = process.memory_info().rss
    rss_mb = rss_after / (1024**2)


    return JSONResponse({
        "topk": results,
        "timing_ms": {
            "inference": round(infer_ms, 2)
        },
        "memory": {
            "rss_mb": round(rss_mb, 1),
            "delta_mb": round((rss_after - rss_before) / (1024**2), 3)
        },
        "threads": process.num_threads()
    })


