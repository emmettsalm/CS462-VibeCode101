# CS-462

Thai Handwritten Numeral Recognition — classes ๕๖ ๕๗ ๕๘ ๕๙ ๖๐

Built with Node.js + TensorFlow.js. Draw a Thai numeral on the canvas and the CNN model predicts which number it is.

## Quick start

```bash
npm install
node server.js
# Open http://localhost:3000
```

## Deploy to Vercel

```bash
npm i -g vercel
vercel
```

## Retrain / replace model (local only)

```bash
python train_model.py --data dataset_thai_v4 --epochs 30
python convert_model.py
node server.js
```
