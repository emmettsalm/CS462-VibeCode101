"""
Convert models/model.h5 → models/tfjs/
Saves weights as binary + shapes JSON (no Keras topology — bypasses Keras 3 incompatibility)
Run: python convert_model.py
"""
import os, json
import numpy as np
import tensorflow as tf

MODEL_H5 = os.path.join('models', 'model.h5')
OUT_DIR  = os.path.join('models', 'tfjs')
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {MODEL_H5} ...")
model = tf.keras.models.load_model(MODEL_H5)

# TF.js model.weights = trainableWeights + nonTrainableWeights (not layer-by-layer)
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
