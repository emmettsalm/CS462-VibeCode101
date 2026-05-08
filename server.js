const express = require('express');
const path    = require('path');
const fs      = require('fs');
const multer  = require('multer');
const tf      = require('@tensorflow/tfjs');
const Jimp    = require('jimp');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '10mb' }));
app.use('/static', express.static(path.join(__dirname, 'static')));

const CLASSES   = ['๕๖', '๕๗', '๕๘', '๕๙', '๖๐'];
const MODEL_DIR     = path.join(__dirname, 'models', 'tfjs');
const WEIGHTS_JSON  = path.join(MODEL_DIR, 'weights.json');
const WEIGHTS_BIN   = path.join(MODEL_DIR, 'group1-shard1of1.bin');

let model = null;

// ── Build CNN architecture (mirrors train_model.py) ───────────────────────
function buildArchitecture() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape: [28,28,1], filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.batchNormalization());
  m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.batchNormalization());
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.5 }));
  m.add(tf.layers.dense({ units: 5, activation: 'softmax' }));
  return m;
}

// ── Model loading ─────────────────────────────────────────────────────────
async function loadModel() {
  if (!fs.existsSync(WEIGHTS_JSON) || !fs.existsSync(WEIGHTS_BIN)) {
    console.warn('[WARN] No TF.js weights found. Run: python convert_model.py');
    return null;
  }
  try {
    const specs   = JSON.parse(fs.readFileSync(WEIGHTS_JSON, 'utf-8'));
    const binBuf  = fs.readFileSync(WEIGHTS_BIN);
    const floats  = new Float32Array(binBuf.buffer, binBuf.byteOffset, binBuf.byteLength / 4);

    // Slice binary into per-weight tensors (positional order matches Keras export)
    let offset = 0;
    const weightTensors = specs.map(spec => {
      const size   = spec.shape.reduce((a, b) => a * b, 1);
      const data   = floats.slice(offset, offset + size);
      offset += size;
      return tf.tensor(data, spec.shape);
    });

    const m = buildArchitecture();
    m.setWeights(weightTensors);
    weightTensors.forEach(t => t.dispose());

    // Warm-up
    const dummy = tf.zeros([1, 28, 28, 1]);
    m.predict(dummy).dispose();
    dummy.dispose();

    model = m;
    console.log('[INFO] Model loaded —', specs.length, 'weight tensors');
  } catch (e) {
    console.error('[ERROR] Model load failed:', e.message);
  }
  return model;
}

// ── Image preprocessing ───────────────────────────────────────────────────
async function preprocessImage(base64Data) {
  const base64 = base64Data.replace(/^data:image\/\w+;base64,/, '');
  const buf    = Buffer.from(base64, 'base64');

  const img = await Jimp.read(buf);
  img.grayscale();

  const { width, height } = img.bitmap;
  const pixels = new Uint8Array(width * height);
  img.scan(0, 0, width, height, (x, y, idx) => {
    pixels[y * width + x] = img.bitmap.data[idx]; // R channel (grayscale)
  });

  // Invert if white background
  const mean = pixels.reduce((s, v) => s + v, 0) / pixels.length;
  if (mean > 127) pixels.forEach((_, i, a) => a[i] = 255 - a[i]);

  // Threshold
  pixels.forEach((v, i, a) => { a[i] = v > 40 ? v : 0; });

  // Bounding box
  let minX = width, minY = height, maxX = 0, maxY = 0, hasContent = false;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (pixels[y * width + x] > 0) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        hasContent = true;
      }
    }
  }

  let cropW = width, cropH = height;
  let cropPixels = pixels;

  if (hasContent) {
    const pad = Math.max(maxX - minX, maxY - minY) >> 2;
    const x1 = Math.max(0, minX - pad), y1 = Math.max(0, minY - pad);
    const x2 = Math.min(width,  maxX + pad + 1);
    const y2 = Math.min(height, maxY + pad + 1);
    cropW = x2 - x1; cropH = y2 - y1;
    cropPixels = new Uint8Array(cropW * cropH);
    for (let y = 0; y < cropH; y++)
      for (let x = 0; x < cropW; x++)
        cropPixels[y * cropW + x] = pixels[(y1 + y) * width + (x1 + x)];
  }

  // Square pad
  const size = Math.max(cropW, cropH);
  const sq   = new Uint8Array(size * size);
  const yOff = (size - cropH) >> 1, xOff = (size - cropW) >> 1;
  for (let y = 0; y < cropH; y++)
    for (let x = 0; x < cropW; x++)
      sq[(yOff + y) * size + (xOff + x)] = cropPixels[y * cropW + x];

  // Resize to 28×28 with Jimp
  const sqImg = new Jimp({ width: size, height: size, color: 0 });
  sqImg.scan(0, 0, size, size, (x, y, idx) => {
    const v = sq[y * size + x];
    sqImg.bitmap.data[idx]     = v;
    sqImg.bitmap.data[idx + 1] = v;
    sqImg.bitmap.data[idx + 2] = v;
    sqImg.bitmap.data[idx + 3] = 255;
  });
  sqImg.resize(28, 28, Jimp.RESIZE_BILINEAR);

  const floats = new Float32Array(28 * 28);
  sqImg.scan(0, 0, 28, 28, (x, y, idx) => {
    floats[y * 28 + x] = sqImg.bitmap.data[idx] / 255;
  });

  return tf.tensor4d(floats, [1, 28, 28, 1]);
}

// ── Routes ────────────────────────────────────────────────────────────────
app.get('/', (_req, res) =>
  res.sendFile(path.join(__dirname, 'templates', 'index.html')));

app.get('/admin', (_req, res) =>
  res.sendFile(path.join(__dirname, 'templates', 'admin.html')));

app.post('/api/predict', async (req, res) => {
  if (!model) return res.status(503).json({ error: 'ยังไม่มีโมเดล กรุณา python convert_model.py แล้วรีสตาร์ท' });

  const { image } = req.body;
  if (!image) return res.status(400).json({ error: 'ไม่พบข้อมูลภาพ' });

  try {
    const tensor = await preprocessImage(image);
    const pred   = model.predict(tensor);
    const probs  = await pred.data();
    tensor.dispose(); pred.dispose();

    const maxIdx = probs.indexOf(Math.max(...probs));
    res.json({
      prediction:    CLASSES[maxIdx],
      confidence:    +probs[maxIdx].toFixed(4),
      probabilities: Object.fromEntries(CLASSES.map((c, i) => [c, +probs[i].toFixed(4)]))
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const upload = multer({ dest: path.join(__dirname, 'models') });
app.post('/api/upload_model', upload.single('model_file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'ไม่พบไฟล์' });

  const ext = path.extname(req.file.originalname).toLowerCase();
  if (!['.h5', '.keras'].includes(ext)) {
    fs.unlinkSync(req.file.path);
    return res.status(400).json({ error: 'ใช้ .h5 หรือ .keras เท่านั้น' });
  }

  const dest = path.join(__dirname, 'models', 'model' + ext);
  fs.renameSync(req.file.path, dest);

  // Convert new model via Python subprocess
  const { execFile } = require('child_process');
  const py = process.platform === 'win32'
    ? 'C:\\Users\\emmet\\AppData\\Local\\Programs\\Python\\Python313\\python.exe'
    : 'python3';

  execFile(py, ['convert_model.py'], { cwd: __dirname }, async (err, stdout, stderr) => {
    if (err) {
      return res.status(500).json({ error: `แปลง model ล้มเหลว: ${err.message}` });
    }
    model = null;
    await loadModel();
    res.json({ message: `อัปโหลดและโหลดโมเดลสำเร็จ (${path.basename(dest)})` });
  });
});

app.get('/api/model_status', (_req, res) =>
  res.json({ loaded: model !== null, classes: CLASSES }));

// ── Start ─────────────────────────────────────────────────────────────────
loadModel().then(() => {
  app.listen(PORT, () => console.log(`Server → http://localhost:${PORT}`));
});
