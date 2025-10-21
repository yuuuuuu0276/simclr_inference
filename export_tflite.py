# export_tflite.py
import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers

# ---- Rebuild your model exactly as in app.py ----
class SimCLRLinearClassifier(tf.keras.Model):
    def __init__(self, simclr_path, num_classes, weight_decay=0.0):
        super().__init__()
        self.encoder = tf.saved_model.load(simclr_path)  # expects 'final_avg_pool'
        self.head = layers.Dense(
            num_classes,
            name="linear_head",
            kernel_regularizer=(tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None)
        )
    def call(self, images, training=False):
        feats = self.encoder(images, trainable=False)['final_avg_pool']
        feats = tf.stop_gradient(feats)
        return self.head(feats, training=training)

simclr_path = "./saved_model"       # folder that has saved_model.pb
num_classes = 40
ckpt_path = "./best_linear_probe3.weights.h5"
IMG_SIZE = (224, 224) 

model = SimCLRLinearClassifier(simclr_path, num_classes)
dummy = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
model.build(IMG_SIZE)
_ = model(dummy, training = False)  # build shapes
model.load_weights(ckpt_path)
_ = model(dummy, training=False)
model.load_weights(ckpt_path)
_ = model(dummy, training=False)


# ----- (1) Dynamic-range quant (no calibration data needed) -----
def convert_dynamic_range(out_path="model_dynamic.tflite"):
    concrete = tf.function(model, input_signature=[tf.TensorSpec([1,224,224,3], tf.float32)])
    concrete_fn = concrete.get_concrete_function(tf.zeros([1,224,224,3], tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(out_path, "wb").write(tflite_model)
    print("Wrote", out_path)

# ----- (2) Full INT8 quant (better size/speed; needs reps) -----
def representative_data_gen(img_dir="./calib_images", n=100):
    # yield n images shaped [1,224,224,3], float32 in [0,1] or whatever your preprocess uses
    import glob
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(img_dir, "*")))[:n]
    if not paths:
        # fallback to random images if you don't have a folder yet
        for _ in range(n):
            yield [np.random.rand(1,224,224,3).astype(np.float32)]
    else:
        for p in paths:
            im = Image.open(p).convert("RGB").resize((224,224))
            x = (np.asarray(im, np.float32)/255.0 - 0.5) * 2.0
            yield [x[None, ...]]

def convert_int8(out_path="model_int8.tflite"):
    concrete = tf.function(model, input_signature=[tf.TensorSpec([1,224,224,3], tf.float32)])
    concrete_fn = concrete.get_concrete_function(tf.zeros([1,224,224,3], tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    open(out_path, "wb").write(tflite_model)
    print("Wrote", out_path)

if __name__ == "__main__":
    convert_dynamic_range("model_dynamic.tflite")
    # For best results, also run the int8 conversion once you have ~50–200 sample images:
    # convert_int8("model_int8.tflite")
