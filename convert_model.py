"""
Convert a Keras .h5 / .keras model → custom TFjs weight format
(weights.json + group1-shard1of1.bin)

Usage:
    python convert_model.py                              # default paths
    python convert_model.py --input model.h5 --output tfjs_out/

Team Members:
    อนาวินธุ์ อักษรทิพย์      1660701440
    ดฤพล กรณ์ถาวรวงศ์        1660703974
    เอ็มเม็ต มีชัย แซลมอน     1660704444
    ธนวัฒน์ วิเศษชัยวรรณ      1660703990
"""
import os, json, argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--input',  default=os.path.join('models', 'model.h5'),
                    help='Path to input .h5 or .keras file')
parser.add_argument('--output', default=os.path.join('models', 'tfjs'),
                    help='Output directory for weights.json + .bin')
args = parser.parse_args()

MODEL_H5 = args.input
OUT_DIR  = args.output
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {MODEL_H5} ...")
model = tf.keras.models.load_model(MODEL_H5)

weights = model.trainable_weights + model.non_trainable_weights
specs   = [{'name': w.name, 'shape': list(w.shape)} for w in weights]
data    = b''.join(w.numpy().astype(np.float32).tobytes() for w in weights)

bin_path  = os.path.join(OUT_DIR, 'group1-shard1of1.bin')
spec_path = os.path.join(OUT_DIR, 'weights.json')

with open(bin_path,  'wb') as f: f.write(data)
with open(spec_path, 'w')  as f: json.dump(specs, f, indent=2)

print(f"[Done] {spec_path}  ({len(specs)} weight tensors)")
print(f"[Done] {bin_path}   ({len(data):,} bytes)")
for s in specs:
    print(f"  {s['name']:40s} {s['shape']}")
