const express  = require('express');
const path     = require('path');
const fs       = require('fs');
const multer   = require('multer');
const tf       = require('@tensorflow/tfjs');
const Jimp     = require('jimp');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '10mb' }));
app.use('/static', express.static(path.join(__dirname, 'static')));

const CLASSES = ['๕๖', '๕๗', '๕๘', '๕๙', '๖๐'];

const DEFAULT_JSON = path.join(__dirname, 'models', 'tfjs', 'weights.json');
const DEFAULT_BIN  = path.join(__dirname, 'models', 'tfjs', 'group1-shard1of1.bin');

let model     = null;
let modelMeta = null;

// ── Architecture ──────────────────────────────────────────────────────────
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

// ── Load model from explicit paths ────────────────────────────────────────
async function loadModelFrom(jsonPath, binPath, sourceName) {
  if (!fs.existsSync(jsonPath) || !fs.existsSync(binPath)) {
    console.warn('[WARN] Weights not found:', jsonPath);
    return false;
  }
  try {
    const specs  = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    const binBuf = fs.readFileSync(binPath);
    const floats = new Float32Array(binBuf.buffer, binBuf.byteOffset, binBuf.byteLength / 4);

    let offset = 0;
    const tensors = specs.map(spec => {
      const size = spec.shape.reduce((a, b) => a * b, 1);
      const data = floats.slice(offset, offset + size);
      offset += size;
      return tf.tensor(data, spec.shape);
    });

    if (model) { model.dispose(); model = null; }
    const m = buildArchitecture();
    m.setWeights(tensors);
    tensors.forEach(t => t.dispose());

    const dummy = tf.zeros([1, 28, 28, 1]);
    m.predict(dummy).dispose();
    dummy.dispose();

    model = m;

    const numParams = specs.reduce((sum, s) => sum + s.shape.reduce((a, b) => a * b, 1), 0);
    modelMeta = {
      source:      sourceName,
      isDefault:   sourceName === 'Default',
      numTensors:  specs.length,
      numParams,
      fileSizeKB:  Math.round(binBuf.byteLength / 1024),
      loadedAt:    new Date().toISOString(),
      classes:     CLASSES,
      inputShape:  '28 × 28 grayscale',
    };

    console.log(`[INFO] Loaded "${sourceName}" — ${specs.length} tensors, ${numParams.toLocaleString()} params`);
    return true;
  } catch (e) {
    console.error('[ERROR] loadModelFrom failed:', e.message);
    return false;
  }
}

async function loadDefaultModel() {
  return loadModelFrom(DEFAULT_JSON, DEFAULT_BIN, 'Default');
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
    pixels[y * width + x] = img.bitmap.data[idx];
  });

  const mean = pixels.reduce((s, v) => s + v, 0) / pixels.length;
  if (mean > 127) pixels.forEach((_, i, a) => { a[i] = 255 - a[i]; });

  pixels.forEach((v, i, a) => { a[i] = v > 40 ? v : 0; });

  let minX = width, minY = height, maxX = 0, maxY = 0, hasContent = false;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      if (pixels[y * width + x] > 0) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        hasContent = true;
      }

  let cropW = width, cropH = height, cropPixels = pixels;
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

  const size = Math.max(cropW, cropH);
  const sq   = new Uint8Array(size * size);
  const yOff = (size - cropH) >> 1, xOff = (size - cropW) >> 1;
  for (let y = 0; y < cropH; y++)
    for (let x = 0; x < cropW; x++)
      sq[(yOff + y) * size + (xOff + x)] = cropPixels[y * cropW + x];

  const sqImg = new Jimp(size, size, 0x000000ff);
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
  if (!model) return res.status(503).json({ error: 'ยังไม่มีโมเดล' });
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

app.get('/api/model_status', (_req, res) =>
  res.json({ loaded: model !== null, meta: modelMeta }));

// ── Upload & convert new model ────────────────────────────────────────────
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

  const { execFile } = require('child_process');
  const py = process.platform === 'win32'
    ? 'C:\\Users\\emmet\\AppData\\Local\\Programs\\Python\\Python313\\python.exe'
    : 'python3';

  execFile(py, ['convert_model.py'], { cwd: __dirname }, async (err) => {
    if (err) return res.status(500).json({ error: `แปลง model ล้มเหลว: ${err.message}` });

    const ok = await loadModelFrom(DEFAULT_JSON, DEFAULT_BIN, req.file.originalname);
    if (!ok) return res.status(500).json({ error: 'โหลดโมเดลใหม่ไม่สำเร็จ' });

    res.json({ message: `โหลด "${req.file.originalname}" สำเร็จ`, meta: modelMeta });
  });
});

// ── Reset to default model ────────────────────────────────────────────────
app.post('/api/reset_model', async (_req, res) => {
  const ok = await loadModelFrom(DEFAULT_JSON, DEFAULT_BIN, 'Default');
  if (!ok) return res.status(500).json({ error: 'โหลดโมเดลเริ่มต้นไม่สำเร็จ' });
  res.json({ message: 'กลับเป็นโมเดลเริ่มต้นแล้ว', meta: modelMeta });
});

// ── Start ─────────────────────────────────────────────────────────────────
loadDefaultModel().then(() => {
  app.listen(PORT, () => console.log(`Server → http://localhost:${PORT}`));
});
