/*
 * Thai Numeral Recognition — Canvas & Predict Logic
 * CS462 Coding Assignment
 *
 * Team Members:
 *   อนาวินธุ์ อักษรทิพย์      1660701440
 *   ดฤพล กรณ์ถาวรวงศ์        1660703974
 *   เอ็มเม็ต มีชัย แซลมอน     1660704444
 *   ธนวัฒน์ วิเศษชัยวรรณ      1660703990
 */
(function () {
  const canvas    = document.getElementById('drawCanvas');
  const ctx       = canvas.getContext('2d');
  const hint      = document.getElementById('canvasHint');
  const wrapper   = document.getElementById('canvasWrapper');
  const brushInput = document.getElementById('brushSize');
  const brushDot   = document.getElementById('brushDot');
  const predictBtn = document.getElementById('predictBtn');
  const clearBtn   = document.getElementById('clearBtn');

  let drawing   = false;
  let hasDrawn  = false;
  let brushRadius = parseInt(brushInput.value, 10);

  function initCanvas() {
    ctx.fillStyle  = '#050510';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap    = 'round';
    ctx.lineJoin   = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth  = brushRadius * 2;
  }
  initCanvas();

  brushInput.addEventListener('input', () => {
    brushRadius = parseInt(brushInput.value, 10);
    ctx.lineWidth = brushRadius * 2;
    if (brushDot) {
      brushDot.style.width  = brushRadius + 'px';
      brushDot.style.height = brushRadius + 'px';
    }
  });

  function getPos(e) {
    const rect   = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const src    = e.touches ? e.touches[0] : e;
    return {
      x: (src.clientX - rect.left) * scaleX,
      y: (src.clientY - rect.top)  * scaleY
    };
  }

  function startDraw(e) {
    e.preventDefault();
    drawing = true;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    if (!hasDrawn) {
      hasDrawn = true;
      hint.classList.add('hidden');
    }
  }

  function draw(e) {
    if (!drawing) return;
    e.preventDefault();
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function stopDraw(e) {
    if (!drawing) return;
    e.preventDefault();
    drawing = false;
    ctx.beginPath();
  }

  canvas.addEventListener('mousedown',  startDraw);
  canvas.addEventListener('mousemove',  draw);
  canvas.addEventListener('mouseup',    stopDraw);
  canvas.addEventListener('mouseleave', stopDraw);
  canvas.addEventListener('touchstart', startDraw, { passive: false });
  canvas.addEventListener('touchmove',  draw,      { passive: false });
  canvas.addEventListener('touchend',   stopDraw,  { passive: false });

  function clearCanvas() {
    initCanvas();
    hasDrawn = false;
    hint.classList.remove('hidden');
    showPlaceholder();
  }
  clearBtn.addEventListener('click', clearCanvas);

  function showPlaceholder() {
    document.getElementById('resultPlaceholder').style.display = '';
    document.getElementById('resultContent').style.display     = 'none';
    document.getElementById('errorContent').style.display      = 'none';
  }

  function showResult(data) {
    document.getElementById('resultPlaceholder').style.display = 'none';
    document.getElementById('errorContent').style.display      = 'none';

    // Re-create node to retrigger animation
    const old = document.getElementById('resultContent');
    const fresh = old.cloneNode(true);
    fresh.style.display = '';
    old.parentNode.replaceChild(fresh, old);

    fresh.querySelector('#predictionText').textContent = data.prediction;
    fresh.querySelector('#confidenceText').textContent =
      `ความมั่นใจ: ${(data.confidence * 100).toFixed(1)}%`;

    const container = fresh.querySelector('#probBars');
    container.innerHTML = '';
    const sorted  = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    const topCls  = sorted[0][0];

    sorted.forEach(([cls, prob]) => {
      const pct   = (prob * 100).toFixed(1);
      const isTop = cls === topCls;
      const row   = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-label">${cls}</span>
        <div class="prob-track">
          <div class="prob-fill ${isTop ? 'top' : ''}" style="width:0%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>`;
      container.appendChild(row);
      // Animate after paint
      requestAnimationFrame(() =>
        requestAnimationFrame(() =>
          row.querySelector('.prob-fill').style.width = pct + '%'));
    });
  }

  function showError(msg) {
    document.getElementById('resultPlaceholder').style.display = 'none';
    document.getElementById('resultContent').style.display     = 'none';
    document.getElementById('errorContent').style.display      = '';
    document.getElementById('errorText').textContent           = msg;
  }

  async function predict() {
    if (!hasDrawn) { showError('กรุณาวาดตัวเลขก่อนกดทำนาย'); return; }

    predictBtn.textContent = 'กำลังทำนาย…';
    predictBtn.classList.add('loading');

    try {
      const res  = await fetch('/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image: canvas.toDataURL('image/png') })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      showResult(data);
    } catch (err) {
      showError(err.message);
    } finally {
      predictBtn.textContent = 'ทำนาย';
      predictBtn.classList.remove('loading');
    }
  }

  predictBtn.addEventListener('click', predict);
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter')                       predict();
    if (e.key === 'Escape' || e.key === 'Delete') clearCanvas();
  });
})();
