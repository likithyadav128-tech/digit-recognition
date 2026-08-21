/* ============================================
   NEURAL DIGIT — Application Logic (Redesigned)
   Auto-trains in background, draw or upload to predict
   ============================================ */

// ---- State ----
let model = null;
let isModelReady = false;
let isDrawing = false;
let lastPos = null;
let history = [];

// ---- DOM ----
const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d');
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    initDragDrop();
    initProbGrid();
    injectRingGradient();
    loadModel();
});

// ---- SVG gradient for ring (injected dynamically) ----
function injectRingGradient() {
    const svg = document.querySelector('.confidence-ring');
    if (!svg) return;
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    grad.id = 'ringGradient';
    grad.setAttribute('x1', '0%'); grad.setAttribute('y1', '0%');
    grad.setAttribute('x2', '100%'); grad.setAttribute('y2', '100%');
    const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    s1.setAttribute('offset', '0%'); s1.setAttribute('stop-color', '#818cf8');
    const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    s2.setAttribute('offset', '100%'); s2.setAttribute('stop-color', '#c084fc');
    grad.appendChild(s1); grad.appendChild(s2);
    defs.appendChild(grad);
    svg.insertBefore(defs, svg.firstChild);
}

// ---- Prob grid ----
function initProbGrid() {
    const grid = document.getElementById('probGrid');
    grid.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span class="prob-label">${i}</span>
            <div class="prob-track"><div class="prob-fill" id="pf${i}" style="width:0%"></div></div>
            <span class="prob-pct" id="pp${i}">0%</span>
        `;
        grid.appendChild(row);
    }
}

// ========== CANVAS ==========
function initCanvas() {
    drawCtx.fillStyle = '#0a0a14';
    drawCtx.fillRect(0, 0, 280, 280);
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
    drawCtx.strokeStyle = '#ffffff';

    drawCanvas.addEventListener('mousedown', startDraw);
    drawCanvas.addEventListener('mousemove', draw);
    drawCanvas.addEventListener('mouseup', endDraw);
    drawCanvas.addEventListener('mouseleave', endDraw);
    drawCanvas.addEventListener('touchstart', wrapTouch(startDraw), { passive: false });
    drawCanvas.addEventListener('touchmove', wrapTouch(draw), { passive: false });
    drawCanvas.addEventListener('touchend', endDraw);
}

function wrapTouch(fn) {
    return function (e) {
        e.preventDefault();
        const t = e.touches[0];
        const r = drawCanvas.getBoundingClientRect();
        const sx = drawCanvas.width / r.width;
        const sy = drawCanvas.height / r.height;
        fn({ offsetX: (t.clientX - r.left) * sx, offsetY: (t.clientY - r.top) * sy });
    };
}

function startDraw(e) {
    isDrawing = true;
    lastPos = getPos(e);
    document.getElementById('canvasHint').classList.add('hidden');
    drawCtx.beginPath();
    drawCtx.arc(lastPos.x, lastPos.y, getBrush() / 2, 0, Math.PI * 2);
    drawCtx.fillStyle = '#fff';
    drawCtx.fill();
}

function draw(e) {
    if (!isDrawing) return;
    const pos = getPos(e);
    drawCtx.lineWidth = getBrush();
    drawCtx.strokeStyle = '#fff';
    drawCtx.beginPath();
    drawCtx.moveTo(lastPos.x, lastPos.y);
    drawCtx.lineTo(pos.x, pos.y);
    drawCtx.stroke();
    lastPos = pos;
}

function endDraw() { isDrawing = false; lastPos = null; }
function getPos(e) { return { x: e.offsetX, y: e.offsetY }; }
function getBrush() { return parseInt(document.getElementById('brushSlider').value); }

function clearCanvas() {
    drawCtx.fillStyle = '#0a0a14';
    drawCtx.fillRect(0, 0, 280, 280);
    document.getElementById('canvasHint').classList.remove('hidden');
    resetResult();
}

// ========== UPLOAD ==========
const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
const ALLOWED_EXT = ['.png', '.jpg', '.jpeg'];

function isValidImageFile(file) {
    if (!file) return false;
    const name = file.name.toLowerCase();
    const typeOk = ALLOWED_TYPES.includes(file.type);
    const extOk = ALLOWED_EXT.some(ext => name.endsWith(ext));
    return typeOk || extOk;
}

function initDragDrop() {
    const zone = document.getElementById('uploadZone');
    ['dragenter', 'dragover'].forEach(ev =>
        zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('drag-over'); })
    );
    ['dragleave', 'drop'].forEach(ev =>
        zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('drag-over'); })
    );
    zone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        if (isValidImageFile(file)) {
            loadUploadedFile(file);
        } else if (file) {
            alert('Please upload a PNG, JPG, or JPEG image.');
        }
    });
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (isValidImageFile(file)) {
        loadUploadedFile(file);
    } else if (file) {
        alert('Please upload a PNG, JPG, or JPEG image.');
        e.target.value = '';
    }
}

function loadUploadedFile(file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = document.getElementById('uploadedImage');
        img.onload = () => {
            if (isModelReady) {
                document.getElementById('predictUploadBtn').disabled = false;
                // Auto recognize when uploaded
                predictUploadedImage();
            }
        };
        img.src = ev.target.result;
        document.getElementById('uploadZone').classList.add('hidden');
        document.getElementById('uploadPreview').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    document.getElementById('uploadZone').classList.remove('hidden');
    document.getElementById('uploadPreview').classList.add('hidden');
    document.getElementById('uploadedImage').src = '';
    document.getElementById('fileInput').value = '';
    resetResult();
}

// ========== TABS ==========
function switchTab(tab) {
    document.getElementById('tabDraw').classList.toggle('active', tab === 'draw');
    document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
    document.getElementById('panelDraw').classList.toggle('hidden', tab !== 'draw');
    document.getElementById('panelUpload').classList.toggle('hidden', tab !== 'upload');
}

// ========== LOAD PRE-TRAINED MODEL ==========
async function loadModel() {
    const overlay = document.getElementById('loadingOverlay');
    const status = document.getElementById('loaderStatus');
    const bar = document.getElementById('loaderBarFill');

    try {
        status.textContent = 'Loading pre-trained CNN (99.2% accuracy)...';
        bar.style.width = '45%';

        // Load pre-trained CNN model
        model = await tf.loadLayersModel('./tfjs_model/model.json?v=6.0');
        bar.style.width = '85%';

        // Warm up inference engine with dummy pass
        status.textContent = 'Warming up inference engine...';
        const dummy = tf.zeros([1, 28, 28, 1]);
        const warmup = model.predict(dummy);
        await warmup.data();
        dummy.dispose();
        warmup.dispose();

        bar.style.width = '100%';
        status.textContent = 'Neural engine ready!';
        isModelReady = true;

        await sleep(200);
        overlay.classList.add('fade-out');
        document.getElementById('mainApp').classList.add('visible');
        document.getElementById('pillDot').classList.add('ready');
        document.getElementById('pillText').textContent = 'Model ready · 99.2% Acc';
        document.getElementById('predictBtn').disabled = false;
        if (!document.getElementById('uploadPreview').classList.contains('hidden')) {
            document.getElementById('predictUploadBtn').disabled = false;
        }
    } catch (err) {
        console.error('Error loading model:', err);
        status.textContent = 'Error: ' + err.message;
    }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ========== PREDICT (DRAW) ==========
async function predictDigit() {
    if (!model || !isModelReady) {
        alert('Engine is still loading, please wait a moment.');
        return;
    }
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    try {
        const tensor = preprocessCanvas(drawCanvas);
        await runPrediction(tensor);
    } catch (err) {
        console.error('Prediction error:', err);
        alert('Prediction error: ' + err.message);
    } finally {
        btn.disabled = false;
    }
}

// ========== FAST SEPARABLE BACKGROUND ESTIMATOR ==========
function computeBackgroundEstimate(src, w, h) {
    const radius = Math.max(10, Math.floor(Math.min(w, h) * 0.12));
    const temp = new Float32Array(w * h);
    const dest = new Float32Array(w * h);

    // Horizontal pass (sliding window O(1) per pixel)
    for (let y = 0; y < h; y++) {
        let sum = 0, count = 0;
        const rowOffset = y * w;
        for (let x = -radius; x <= radius; x++) {
            if (x >= 0 && x < w) {
                sum += src[rowOffset + x];
                count++;
            }
        }
        temp[rowOffset] = sum / count;

        for (let x = 1; x < w; x++) {
            const addX = x + radius;
            const remX = x - radius - 1;
            if (addX < w) { sum += src[rowOffset + addX]; count++; }
            if (remX >= 0) { sum -= src[rowOffset + remX]; count--; }
            temp[rowOffset + x] = sum / count;
        }
    }

    // Vertical pass (sliding window O(1) per pixel)
    for (let x = 0; x < w; x++) {
        let sum = 0, count = 0;
        for (let y = -radius; y <= radius; y++) {
            if (y >= 0 && y < h) {
                sum += temp[y * w + x];
                count++;
            }
        }
        dest[x] = sum / count;

        for (let y = 1; y < h; y++) {
            const addY = y + radius;
            const remY = y - radius - 1;
            if (addY < h) { sum += temp[addY * w + x]; count++; }
            if (remY >= 0) { sum -= temp[remY * w + x]; count--; }
            dest[y * w + x] = sum / count;
        }
    }

    return dest;
}

// ========== PREDICT (UPLOAD) ==========
async function predictUploadedImage() {
    if (!model || !isModelReady) {
        alert('Engine is still loading, please wait a moment.');
        return;
    }

    const btn = document.getElementById('predictUploadBtn');
    btn.disabled = true;

    try {
        const img = document.getElementById('uploadedImage');
        if (!img || !img.src) {
            alert('Please select or upload an image first.');
            return;
        }

        // Downscale large camera photos to max 350px for fast, sharp processing
        const maxDim = 350;
        let w = img.naturalWidth || img.width || 280;
        let h = img.naturalHeight || img.height || 280;
        if (Math.max(w, h) > maxDim) {
            const scale = maxDim / Math.max(w, h);
            w = Math.round(w * scale);
            h = Math.round(h * scale);
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = w;
        tempCanvas.height = h;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);

        const imgData = ctx.getImageData(0, 0, w, h);

        // --- Step 1: Grayscale conversion ---
        const gray = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
            gray[i] = 0.299 * imgData.data[i * 4] + 0.587 * imgData.data[i * 4 + 1] + 0.114 * imgData.data[i * 4 + 2];
        }

        // --- Step 2: Background Estimation & Shadow Cancellation ---
        // This removes all camera shadows, uneven room lighting, and paper creases
        const bg = computeBackgroundEstimate(gray, w, h);

        // Compute local brightness to handle both dark-on-light and light-on-dark
        let avgBg = 0;
        for (let i = 0; i < bg.length; i++) avgBg += bg[i];
        avgBg /= bg.length;

        const isLightPaper = avgBg > 100;
        const ink = new Float32Array(w * h);
        let maxInk = 0;

        for (let i = 0; i < w * h; i++) {
            // Subtract local background to extract ONLY the ink
            const diff = isLightPaper ? (bg[i] - gray[i]) : (gray[i] - bg[i]);
            if (diff > 0) {
                ink[i] = diff;
                if (diff > maxInk) maxInk = diff;
            }
        }

        // --- Step 3: Adaptive Ink Thresholding ---
        // Pen strokes have high difference relative to flat paper
        const thresh = Math.max(15, maxInk * 0.25);
        for (let i = 0; i < ink.length; i++) {
            if (ink[i] < thresh) {
                ink[i] = 0;
            } else {
                // Contrast stretch ink strokes
                ink[i] = ((ink[i] - thresh) / (maxInk - thresh)) * 255.0;
            }
        }

        // --- Step 4: Morphological Dilation (thickens thin pencil/pen lines) ---
        const dilated = new Float32Array(w * h);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let maxVal = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                            if (ink[ny * w + nx] > maxVal) maxVal = ink[ny * w + nx];
                        }
                    }
                }
                dilated[y * w + x] = maxVal;
            }
        }

        // --- Step 5: Center-of-mass centering into 28x28 tensor ---
        const tensor = centerByMassAndFit(dilated, w, h);
        await runPrediction(tensor);
    } catch (err) {
        console.error('Upload prediction error:', err);
        alert('Prediction error: ' + err.message);
    } finally {
        btn.disabled = false;
    }
}

// ========== DOMINANT DIGIT & CONNECTED COMPONENT EXTRACTION ==========
function extractDominantDigit(binaryArr, w, h) {
    // 1. Zero out border margins (outer 4% of width/height) to remove card edge/table borders
    const marginX = Math.max(3, Math.floor(w * 0.04));
    const marginY = Math.max(3, Math.floor(h * 0.04));
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (x < marginX || x >= w - marginX || y < marginY || y >= h - marginY) {
                binaryArr[y * w + x] = 0;
            }
        }
    }

    // 2. Connected Component Labeling via BFS
    const labels = new Int32Array(w * h);
    let currentLabel = 0;
    const components = [];

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const idx = y * w + x;
            if (binaryArr[idx] > 20 && labels[idx] === 0) {
                currentLabel++;
                let mass = 0;
                let minX = x, maxX = x, minY = y, maxY = y;
                const queue = [idx];
                labels[idx] = currentLabel;

                while (queue.length > 0) {
                    const curr = queue.pop();
                    const cy = Math.floor(curr / w);
                    const cx = curr % w;
                    mass += binaryArr[curr];

                    if (cx < minX) minX = cx;
                    if (cx > maxX) maxX = cx;
                    if (cy < minY) minY = cy;
                    if (cy > maxY) maxY = cy;

                    const neighbors = [
                        cy > 0 ? (cy - 1) * w + cx : -1,
                        cy < h - 1 ? (cy + 1) * w + cx : -1,
                        cx > 0 ? cy * w + (cx - 1) : -1,
                        cx < w - 1 ? cy * w + (cx + 1) : -1
                    ];

                    for (const n of neighbors) {
                        if (n >= 0 && binaryArr[n] > 20 && labels[n] === 0) {
                            labels[n] = currentLabel;
                            queue.push(n);
                        }
                    }
                }

                components.push({ label: currentLabel, mass, minX, maxX, minY, maxY });
            }
        }
    }

    if (components.length === 0) return null;

    // Find the largest component (the main digit)
    components.sort((a, b) => b.mass - a.mass);
    const mainComp = components[0];

    // Keep all components that are part of the digit (multi-stroke digits like 4, 5, 7)
    const cleanArr = new Float32Array(w * h);
    let finalMinX = mainComp.minX, finalMaxX = mainComp.maxX;
    let finalMinY = mainComp.minY, finalMaxY = mainComp.maxY;

    // Allow nearby strokes within 25% of image dimension
    const maxDist = Math.max(25, Math.floor(Math.min(w, h) * 0.25));

    for (const comp of components) {
        const isSignificant = comp.mass >= mainComp.mass * 0.08;
        const isNearby = (comp.minX <= mainComp.maxX + maxDist && comp.maxX >= mainComp.minX - maxDist &&
                          comp.minY <= mainComp.maxY + maxDist && comp.maxY >= mainComp.minY - maxDist);

        if (isSignificant && isNearby) {
            if (comp.minX < finalMinX) finalMinX = comp.minX;
            if (comp.maxX > finalMaxX) finalMaxX = comp.maxX;
            if (comp.minY < finalMinY) finalMinY = comp.minY;
            if (comp.maxY > finalMaxY) finalMaxY = comp.maxY;

            for (let i = 0; i < w * h; i++) {
                if (labels[i] === comp.label) {
                    cleanArr[i] = binaryArr[i];
                }
            }
        }
    }

    return {
        arr: cleanArr,
        minX: finalMinX, maxX: finalMaxX,
        minY: finalMinY, maxY: finalMaxY
    };
}

// ========== CENTER-OF-MASS PREPROCESSING (shared) ==========
function centerByMassAndFit(grayArr, w, h) {
    const digitData = extractDominantDigit(grayArr, w, h);
    if (!digitData) {
        previewCtx.fillStyle = '#000';
        previewCtx.fillRect(0, 0, 28, 28);
        return tf.tensor4d(new Float32Array(784), [1, 28, 28, 1]);
    }

    const { arr, minX, maxX, minY, maxY } = digitData;
    const cw = Math.max(1, maxX - minX + 1);
    const ch = Math.max(1, maxY - minY + 1);

    // Crop to digit bounding box
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cw; cropCanvas.height = ch;
    const cropCtx = cropCanvas.getContext('2d');
    const cropData = cropCtx.createImageData(cw, ch);
    for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
            const val = Math.round(arr[(y + minY) * w + (x + minX)]);
            const idx = (y * cw + x) * 4;
            cropData.data[idx] = val;
            cropData.data[idx + 1] = val;
            cropData.data[idx + 2] = val;
            cropData.data[idx + 3] = 255;
        }
    }
    cropCtx.putImageData(cropData, 0, 0);

    // Scale to fit 20x20 box (standard MNIST dimension)
    const scale = Math.min(20.0 / cw, 20.0 / ch);
    const newW = Math.max(2, Math.round(cw * scale));
    const newH = Math.max(2, Math.round(ch * scale));

    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = newW; resizeCanvas.height = newH;
    const resizeCtx = resizeCanvas.getContext('2d');
    resizeCtx.imageSmoothingEnabled = true;
    resizeCtx.imageSmoothingQuality = 'high';
    resizeCtx.drawImage(cropCanvas, 0, 0, newW, newH);

    const resizedData = resizeCtx.getImageData(0, 0, newW, newH);
    const resizedGray = new Float32Array(newW * newH);
    for (let i = 0; i < newW * newH; i++) {
        resizedGray[i] = resizedData.data[i * 4];
    }

    // Compute Center of Mass
    let totalMass = 0, comX = 0, comY = 0;
    for (let y = 0; y < newH; y++) {
        for (let x = 0; x < newW; x++) {
            const v = resizedGray[y * newW + x];
            totalMass += v;
            comX += x * v;
            comY += y * v;
        }
    }

    if (totalMass === 0) {
        previewCtx.fillStyle = '#000';
        previewCtx.fillRect(0, 0, 28, 28);
        return tf.tensor4d(new Float32Array(784), [1, 28, 28, 1]);
    }

    comX /= totalMass;
    comY /= totalMass;

    // Shift center of mass to (14, 14)
    const shiftX = Math.round(14.0 - comX);
    const shiftY = Math.round(14.0 - comY);

    const raw28 = new Float32Array(28 * 28);
    for (let y = 0; y < newH; y++) {
        for (let x = 0; x < newW; x++) {
            const ny = y + shiftY;
            const nx = x + shiftX;
            if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
                raw28[ny * 28 + nx] = resizedGray[y * newW + x];
            }
        }
    }

    // Stroke Thickening (matches MNIST brush thickness so thin pen lines are thick and clear)
    const final28 = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let maxV = raw28[y * 28 + x];
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
                        const factor = (dx === 0 && dy === 0) ? 1.0 : 0.65;
                        const v = raw28[ny * 28 + nx] * factor;
                        if (v > maxV) maxV = v;
                    }
                }
            }
            final28[y * 28 + x] = maxV;
        }
    }

    // Render 28x28 preview
    const tmp = document.createElement('canvas');
    tmp.width = 28; tmp.height = 28;
    const tctx = tmp.getContext('2d');
    const previewData = tctx.createImageData(28, 28);

    let maxVal = 0;
    for (let i = 0; i < 784; i++) {
        if (final28[i] > maxVal) maxVal = final28[i];
    }

    for (let i = 0; i < 784; i++) {
        const norm = maxVal > 0 ? (final28[i] / maxVal) * 255.0 : 0;
        const v = Math.round(norm);
        previewData.data[i * 4] = v;
        previewData.data[i * 4 + 1] = v;
        previewData.data[i * 4 + 2] = v;
        previewData.data[i * 4 + 3] = 255;
    }
    tctx.putImageData(previewData, 0, 0);
    previewCtx.drawImage(tmp, 0, 0);

    const tensor = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
        tensor[i] = maxVal > 0 ? final28[i] / maxVal : 0;
    }

    return tf.tensor4d(tensor, [1, 28, 28, 1]);
}

// ========== PREPROCESSING (canvas drawings) ==========
function preprocessCanvas(sourceCanvas) {
    const sw = sourceCanvas.width;
    const sh = sourceCanvas.height;
    const srcCtx = sourceCanvas.getContext('2d');
    const imgData = srcCtx.getImageData(0, 0, sw, sh);
    const d = imgData.data;

    // Convert to grayscale (max of RGB channels — white strokes)
    const gray = new Float32Array(sw * sh);
    for (let i = 0; i < sw * sh; i++) {
        gray[i] = Math.max(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]);
    }

    return centerByMassAndFit(gray, sw, sh);
}

// ========== RUN PREDICTION ==========
async function runPrediction(tensor) {
    const pred = model.predict(tensor);
    const probs = await pred.data();

    let maxP = 0, maxI = 0;
    for (let i = 0; i < 10; i++) {
        if (probs[i] > maxP) { maxP = probs[i]; maxI = i; }
    }

    // Show result
    document.getElementById('resultEmpty').classList.add('hidden');
    const rp = document.getElementById('resultPrediction');
    rp.classList.remove('hidden');
    // Re-trigger animation
    rp.style.animation = 'none';
    void rp.offsetWidth;
    rp.style.animation = '';

    // Digit
    document.getElementById('ringDigit').textContent = maxI;

    // Confidence ring
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference * (1 - maxP);
    document.getElementById('ringFill').style.strokeDashoffset = offset;

    // Confidence text
    const confPct = (maxP * 100).toFixed(1);
    document.getElementById('confValue').textContent = confPct + '%';

    // Prob bars
    for (let i = 0; i < 10; i++) {
        const pct = (probs[i] * 100).toFixed(1);
        const fill = document.getElementById(`pf${i}`);
        const pctEl = document.getElementById(`pp${i}`);
        fill.style.width = pct + '%';
        pctEl.textContent = pct + '%';
        fill.classList.toggle('top', i === maxI);
    }

    // History
    addHistory(maxI, confPct);

    tensor.dispose();
    pred.dispose();
}

function resetResult() {
    document.getElementById('resultEmpty').classList.remove('hidden');
    document.getElementById('resultPrediction').classList.add('hidden');
    previewCtx.fillStyle = '#000';
    previewCtx.fillRect(0, 0, 28, 28);
    for (let i = 0; i < 10; i++) {
        document.getElementById(`pf${i}`).style.width = '0%';
        document.getElementById(`pf${i}`).classList.remove('top');
        document.getElementById(`pp${i}`).textContent = '0%';
    }
}

// ========== HISTORY ==========
function addHistory(digit, conf) {
    history.unshift({ digit, conf });
    if (history.length > 12) history.pop();
    const container = document.getElementById('historyItems');
    container.innerHTML = '';
    history.forEach(h => {
        const chip = document.createElement('div');
        chip.className = 'history-chip';
        chip.innerHTML = `<span class="chip-digit">${h.digit}</span><span class="chip-conf">${h.conf}%</span>`;
        container.appendChild(chip);
    });
}
