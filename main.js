// ============================================================
//  GESTURA — Browser-native, zero-backend edition
//  MediaPipe Tasks Vision JS (same model as Python, runs in WASM)
// ============================================================

import { HandLandmarker, FilesetResolver } from
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- DOM refs ----------------------------------------------------------------
const shapeText      = document.getElementById('shape-text');
const handStateBadge = document.getElementById('hand-state-badge');
const loadingOverlay = document.getElementById('loading-overlay');
const loaderFill     = document.getElementById('loader-fill');
const loaderSubtitle = document.getElementById('loader-subtitle');
const loaderError    = document.getElementById('loader-error');
const cameraCanvas   = document.getElementById('camera-canvas');
const camCtx         = cameraCanvas.getContext('2d');
const trailCanvas    = document.getElementById('trail-canvas');
const trailCtx       = trailCanvas.getContext('2d');
const debugPanel     = document.getElementById('debug-panel');
const videoEl        = document.getElementById('input-video');

// ---- Hand skeleton connections (MediaPipe 21-landmark format) -----------------
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [5,6],[6,7],[7,8],
    [9,10],[10,11],[11,12],
    [13,14],[14,15],[15,16],
    [17,18],[18,19],[19,20],
    [0,5],[5,9],[9,13],[13,17],[0,17]
];

// ---- Gesture state -----------------------------------------------------------
let fingerPos    = new THREE.Vector2(0, 0);
let palmPos      = new THREE.Vector2(0, 0);
let isDrawing    = false;
let handState    = "other";
let currentShape = "random";
let historyShapes = [];
let lastShapeTime = 0;
let drawingPoints = [];  // pixel coords (mirrored, matching what user sees)
let isPinching    = false; // latched state with hysteresis

// Pinch thresholds (normalized 0-1 MediaPipe coords)
// Turn ON when distance < PINCH_ON, stay on until > PINCH_OFF
const PINCH_ON  = 0.07;
const PINCH_OFF = 0.12;

// ============================================================
//  THREE.JS SCENE
// ============================================================
const NUM_PARTICLES = 10000;
const PARTICLE_SIZE = 0.08;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050505, 0.02);

const threeCamera = new THREE.PerspectiveCamera(
    75, window.innerWidth / window.innerHeight, 0.1, 1000
);
threeCamera.position.z = 8;

const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance"
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.domElement.id = 'threejs-canvas'; // CSS z-index hook
document.body.appendChild(renderer.domElement);

// ---- Particle geometry -------------------------------------------------------
const geo          = new THREE.BufferGeometry();
const positions    = new Float32Array(NUM_PARTICLES * 3);
const targets      = new Float32Array(NUM_PARTICLES * 3);
const colors       = new Float32Array(NUM_PARTICLES * 3);
const targetColors = new Float32Array(NUM_PARTICLES * 3);

// Vibrant color palettes per shape
const PALETTES = {
    random:    [0x00f2fe, 0x4facfe, 0xff0844, 0xffb199, 0xf5576c],
    square:    [0x00c6ff, 0x0072ff, 0x4facfe, 0x00f2fe],
    triangle:  [0x00b09b, 0x96c93d, 0x0ba360, 0x3cba92],
    circle:    [0xb224ef, 0x7579ff, 0x667eea, 0x764ba2],
    rectangle: [0x43e97b, 0x38f9d7, 0x00cdac, 0x78ffd6],
    house:     [0xf6d365, 0xfda085, 0xffd194, 0x70e1f5]
};

function getColor(paletteName) {
    const pal = PALETTES[paletteName] || PALETTES.random;
    return new THREE.Color(pal[Math.floor(Math.random() * pal.length)]);
}

// Initialize random particle positions
for (let i = 0; i < NUM_PARTICLES; i++) {
    const x = (Math.random() - 0.5) * 20;
    const y = (Math.random() - 0.5) * 20;
    const z = (Math.random() - 0.5) * 10;
    positions[i*3] = x;   positions[i*3+1] = y;   positions[i*3+2] = z;
    targets  [i*3] = x;   targets  [i*3+1] = y;   targets  [i*3+2] = z;
    const c = getColor('random');
    colors      [i*3] = c.r; colors      [i*3+1] = c.g; colors      [i*3+2] = c.b;
    targetColors[i*3] = c.r; targetColors[i*3+1] = c.g; targetColors[i*3+2] = c.b;
}

geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geo.setAttribute('color',    new THREE.BufferAttribute(colors,    3));

// Soft glow particle texture
const ptCvs  = document.createElement('canvas');
ptCvs.width  = ptCvs.height = 32;
const ptCtx  = ptCvs.getContext('2d');
const ptGrad = ptCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
ptGrad.addColorStop(0, 'rgba(255,255,255,1)');
ptGrad.addColorStop(1, 'rgba(255,255,255,0)');
ptCtx.fillStyle = ptGrad;
ptCtx.fillRect(0, 0, 32, 32);
const ptTex = new THREE.CanvasTexture(ptCvs);

const mat = new THREE.PointsMaterial({
    size: PARTICLE_SIZE,
    vertexColors: true,
    map: ptTex,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    transparent: true,
    opacity: 0.85
});

const particles = new THREE.Points(geo, mat);
scene.add(particles);

// ============================================================
//  SHAPE GEOMETRY GENERATORS  (identical to Python backend)
// ============================================================
function getSquarePoint() {
    const s = 5, e = Math.floor(Math.random()*4), t = Math.random()*s - s/2;
    if (e===0) return {x:t,  y: s/2};
    if (e===1) return {x:t,  y:-s/2};
    if (e===2) return {x: s/2, y:t};
    return {x:-s/2, y:t};
}
function getRectPoint() {
    const W=7,H=4,e=Math.floor(Math.random()*4);
    if (e===0) return {x:Math.random()*W-W/2, y: H/2};
    if (e===1) return {x:Math.random()*W-W/2, y:-H/2};
    if (e===2) return {x: W/2, y:Math.random()*H-H/2};
    return {x:-W/2, y:Math.random()*H-H/2};
}
function getTriPoint() {
    const s=5,e=Math.floor(Math.random()*3),t=Math.random();
    if (e===0) return {x:t*(-s/2)+(1-t)*0,  y:t*(-s/2)+(1-t)*(s/2)};
    if (e===1) return {x:t*(s/2)+(1-t)*0,   y:t*(-s/2)+(1-t)*(s/2)};
    return {x:t*(s/2)+(1-t)*(-s/2), y:-s/2};
}
function getCirclePoint() {
    const a = Math.random()*Math.PI*2, r = 2.5+(Math.random()-0.5)*0.5;
    return {x:Math.cos(a)*r, y:Math.sin(a)*r};
}
function getHousePoint() {
    const s=4;
    if (Math.random()<0.6) {
        const e=Math.floor(Math.random()*3), t=Math.random()*s-s/2;
        if (e===0) return {x:t,    y:-s/2};
        if (e===1) return {x:-s/2, y:t};
        return {x:s/2, y:t};
    }
    const t=Math.random(), e=Math.floor(Math.random()*2);
    if (e===0) return {x:t*(-s/2-0.5)+(1-t)*0, y:t*(s/2)+(1-t)*(s/2+2)};
    return {x:t*(s/2+0.5)+(1-t)*0, y:t*(s/2)+(1-t)*(s/2+2)};
}

function updateTargets(shape) {
    for (let i = 0; i < NUM_PARTICLES; i++) {
        let p = {x:(Math.random()-0.5)*20, y:(Math.random()-0.5)*20};
        if (shape==='square')    p = getSquarePoint();
        else if (shape==='rectangle') p = getRectPoint();
        else if (shape==='triangle')  p = getTriPoint();
        else if (shape==='circle')    p = getCirclePoint();
        else if (shape==='house')     p = getHousePoint();

        const noise = (shape==='random') ? 0 : (Math.random()-0.5)*0.4;
        targets[i*3]   = p.x + noise;
        targets[i*3+1] = p.y + noise;
        targets[i*3+2] = (Math.random()-0.5)*1.0;

        const c = getColor(shape);
        targetColors[i*3]   = c.r;
        targetColors[i*3+1] = c.g;
        targetColors[i*3+2] = c.b;
    }
}

// ============================================================
//  AUDIO
// ============================================================
let audioCtx = null;
function initAudio() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
}
function playSuccessSound() {
    initAudio();
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(440, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(880,  audioCtx.currentTime + 0.1);
    osc.frequency.exponentialRampToValueAtTime(1108, audioCtx.currentTime + 0.2);
    gain.gain.setValueAtTime(0, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0.3, audioCtx.currentTime + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.6);
    osc.start(audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.6);
}
function playAbsorbSound() {
    initAudio();
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(400, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(40, audioCtx.currentTime + 0.3);
    gain.gain.setValueAtTime(0, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0.4, audioCtx.currentTime + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
    osc.start(audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.3);
}
window.addEventListener('click', initAudio);

// ============================================================
//  SHAPE DETECTION  (Douglas-Peucker, equivalent to cv2.approxPolyDP)
// ============================================================
function perpDist(pt, a, b) {
    const dx = b[0]-a[0], dy = b[1]-a[1];
    const len = Math.sqrt(dx*dx + dy*dy);
    if (len < 1e-6) return Math.sqrt((pt[0]-a[0])**2 + (pt[1]-a[1])**2);
    return Math.abs(dy*pt[0] - dx*pt[1] + b[0]*a[1] - b[1]*a[0]) / len;
}

function rdp(pts, eps) {
    if (pts.length < 3) return pts;
    let maxD = 0, maxI = 1;
    for (let i = 1; i < pts.length - 1; i++) {
        const d = perpDist(pts[i], pts[0], pts[pts.length-1]);
        if (d > maxD) { maxD = d; maxI = i; }
    }
    if (maxD > eps) {
        const L = rdp(pts.slice(0, maxI+1), eps);
        const R = rdp(pts.slice(maxI),      eps);
        return [...L.slice(0, -1), ...R];
    }
    return [pts[0], pts[pts.length-1]];
}

function arcLengthClosed(pts) {
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
        const dx = pts[i][0]-pts[i-1][0], dy = pts[i][1]-pts[i-1][1];
        total += Math.sqrt(dx*dx + dy*dy);
    }
    // Add closing segment (last → first)
    const dx = pts[0][0]-pts[pts.length-1][0], dy = pts[0][1]-pts[pts.length-1][1];
    return total + Math.sqrt(dx*dx + dy*dy);
}

function detectShape(pts) {
    if (pts.length < 8) return null;

    const perim = arcLengthClosed(pts);
    if (perim < 80) return null; // Too small a gesture (pixels)

    const eps = 0.04 * perim;

    // Close the polyline (same as OpenCV's closed=True)
    const closed = [...pts, pts[0]];
    const approx = rdp(closed, eps);
    const verts  = approx.length - 1; // -1 because first ≡ last

    if (verts === 3) return 'triangle';
    if (verts === 4) {
        const xs = approx.map(p => p[0]);
        const ys = approx.map(p => p[1]);
        const w  = Math.max(...xs) - Math.min(...xs);
        const h  = Math.max(...ys) - Math.min(...ys);
        const ar = w / (h || 1);
        return (ar >= 0.55 && ar <= 1.8) ? 'square' : 'rectangle';
    }
    if (verts > 6) return 'circle';
    return null;
}

// ============================================================
//  MEDIAPIPE INIT
// ============================================================
let handLandmarker = null;
let lastVideoTime  = -1;

async function initApp() {
    setProgress(15, "Cargando MediaPipe (WASM)...");
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        setProgress(55, "Cargando modelo de mano...");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence:  0.5,
            minTrackingConfidence:      0.5
        });
    } catch (e) {
        showError("No se pudo cargar MediaPipe.<br>" + e.message +
                  "<br><br>Asegúrate de estar en un servidor local (no file://).");
        return;
    }

    setProgress(80, "Solicitando permiso de cámara...");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: "user" }
        });
        videoEl.srcObject = stream;
        await new Promise(res => { videoEl.onloadeddata = res; });
        videoEl.play();
    } catch (e) {
        showError("No se pudo acceder a la cámara.<br>" + e.message);
        return;
    }

    setProgress(100, "¡Listo!");
    setTimeout(() => {
        loadingOverlay.classList.add('fade-out');
        setTimeout(() => { loadingOverlay.style.display = 'none'; }, 650);
    }, 350);

    shapeText.innerText = "¡Pellizca para dibujar!";
    requestAnimationFrame(detectionLoop);
}

function setProgress(pct, msg) {
    loaderFill.style.width = pct + '%';
    loaderSubtitle.textContent = msg;
}

function showError(html) {
    loaderError.style.display = 'block';
    loaderError.innerHTML = html;
    loaderFill.style.background = '#f5576c';
}

// ============================================================
//  GESTURE DETECTION LOOP
// ============================================================
function dist2D(a, b) {
    return Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2);
}

// Helper: mirror x in [0,1] for selfie view
const mx = x => 1 - x;

function resizeCanvases() {
    const W = window.innerWidth, H = window.innerHeight;
    if (cameraCanvas.width  !== W) { cameraCanvas.width  = trailCanvas.width  = W; }
    if (cameraCanvas.height !== H) { cameraCanvas.height = trailCanvas.height = H; }
}

function detectionLoop() {
    requestAnimationFrame(detectionLoop);

    if (!handLandmarker || videoEl.readyState < 2) return;
    resizeCanvases();

    const W = cameraCanvas.width, H = cameraCanvas.height;

    // --- Draw mirrored camera feed on the background canvas ---
    camCtx.clearRect(0, 0, W, H);
    camCtx.save();
    camCtx.translate(W, 0);
    camCtx.scale(-1, 1);
    camCtx.drawImage(videoEl, 0, 0, W, H);
    camCtx.restore();

    // --- MediaPipe detection ---
    // CRITICAL: use performance.now() NOT the RAF timestamp
    const timestampMs = performance.now();
    if (videoEl.currentTime === lastVideoTime) return;
    lastVideoTime = videoEl.currentTime;

    const result = handLandmarker.detectForVideo(videoEl, timestampMs);

    let newHandState  = "other";
    let nowPinching   = false;

    if (result.landmarks && result.landmarks.length > 0) {
        const lm = result.landmarks[0];

        // --- Draw hand skeleton (mirrored) ---
        camCtx.strokeStyle = 'rgba(180,220,255,0.75)';
        camCtx.lineWidth   = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
            camCtx.beginPath();
            camCtx.moveTo(mx(lm[a].x)*W, lm[a].y*H);
            camCtx.lineTo(mx(lm[b].x)*W, lm[b].y*H);
            camCtx.stroke();
        }

        // Key landmarks
        const thumbTip  = lm[4];
        const indexTip  = lm[8];
        const middleTip = lm[12];
        const ringTip   = lm[16];
        const pinkyTip  = lm[20];
        const wrist     = lm[0];
        const palmBase  = lm[9];

        // Map to Three.js world coords (mirrored x so it matches what user sees)
        const fxN = (mx(indexTip.x) + mx(thumbTip.x)) / 2;
        const fyN = (indexTip.y     + thumbTip.y)      / 2;
        fingerPos.set((fxN - 0.5) * 20, -(fyN - 0.5) * 15);
        palmPos.set((mx(palmBase.x) - 0.5) * 20, -(palmBase.y - 0.5) * 15);

        // --- Pinch detection with hysteresis ---
        const pinchDist = dist2D(thumbTip, indexTip);

        if (!isPinching && pinchDist < PINCH_ON)  isPinching = true;
        if (isPinching  && pinchDist > PINCH_OFF) isPinching = false;

        // --- Hand open/closed ---
        const avgDist = (
            dist2D(indexTip, wrist) + dist2D(middleTip, wrist) +
            dist2D(ringTip,  wrist) + dist2D(pinkyTip,  wrist)
        ) / 4;

        if (isPinching) {
            nowPinching  = true;
            newHandState = "pinching";
            initAudio(); // unlock audio on first gesture

            // Collect drawing point in mirrored pixel coords
            const midX = fxN * W;
            const midY = fyN * H;
            drawingPoints.push([midX, midY]);

            // Highlight pinch on camera canvas
            const ix = mx(indexTip.x)*W, iy = indexTip.y*H;
            const tx = mx(thumbTip.x)*W, ty = thumbTip.y*H;
            camCtx.strokeStyle = '#00ff99';
            camCtx.lineWidth = 4;
            camCtx.beginPath(); camCtx.moveTo(ix, iy); camCtx.lineTo(tx, ty); camCtx.stroke();
            dot(camCtx, ix, iy, 12, '#00ff99');
            dot(camCtx, tx, ty, 12, '#00ff99');
        } else {
            const ix = mx(indexTip.x)*W, iy = indexTip.y*H;
            const tx = mx(thumbTip.x)*W, ty = thumbTip.y*H;
            dot(camCtx, ix, iy, 8, '#ff4444');
            dot(camCtx, tx, ty, 8, '#ff4444');
            newHandState = (avgDist < 0.2) ? 'closed' : 'open';
        }

        // All landmark dots
        for (const lmk of lm) {
            dot(camCtx, mx(lmk.x)*W, lmk.y*H, 3.5, 'rgba(80,160,255,0.85)');
        }

        // Debug info
        const marker = isPinching ? '🟢 PINCH' : '⚪';
        debugPanel.textContent =
            `${marker}  dist=${pinchDist.toFixed(3)}  pts=${drawingPoints.length}  state=${newHandState}`;

    } else {
        debugPanel.textContent = '⚠️  No se detecta mano en cámara';
    }

    // --- Shape detection when pinch released ---
    if (isDrawing && !nowPinching) {
        if (drawingPoints.length >= 8) {
            const shape = detectShape(drawingPoints);
            if (shape) {
                const t = performance.now() / 1000;
                if (shape === 'triangle' &&
                    historyShapes.includes('square') &&
                    (t - lastShapeTime) < 5.0) {
                    applyShape('house');
                    historyShapes = [];
                } else {
                    applyShape(shape);
                    historyShapes.push(shape);
                    lastShapeTime = t;
                    if (historyShapes.length > 5) historyShapes.shift();
                }
            }
        }
        // Clear trail after gesture
        trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
        drawingPoints = [];
    }

    // --- Draw current trail (crisp yellow stroke on separate canvas) ---
    trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
    if (drawingPoints.length > 1) {
        // Glowing trail effect
        trailCtx.shadowColor  = 'rgba(255,255,100,0.6)';
        trailCtx.shadowBlur   = 12;
        trailCtx.strokeStyle  = 'rgba(255, 240, 0, 0.95)';
        trailCtx.lineWidth    = 4;
        trailCtx.lineJoin     = 'round';
        trailCtx.lineCap      = 'round';
        trailCtx.beginPath();
        trailCtx.moveTo(drawingPoints[0][0], drawingPoints[0][1]);
        for (let i = 1; i < drawingPoints.length; i++) {
            // Smooth with midpoints
            const mx2 = (drawingPoints[i][0] + drawingPoints[i-1][0]) / 2;
            const my2 = (drawingPoints[i][1] + drawingPoints[i-1][1]) / 2;
            trailCtx.quadraticCurveTo(drawingPoints[i-1][0], drawingPoints[i-1][1], mx2, my2);
        }
        trailCtx.stroke();
        trailCtx.shadowBlur = 0;
    }

    // --- Audio + badge update ---
    if (newHandState === 'closed' && handState !== 'closed' && currentShape !== 'random') {
        playAbsorbSound();
    }

    isDrawing = nowPinching;
    handState = newHandState;

    const LABELS = {
        pinching: '✍️  Dibujando',
        open:     '🖐  Mano abierta',
        closed:   '✊  Puño cerrado',
        other:    '—'
    };
    handStateBadge.textContent = LABELS[handState] || '—';
}

function dot(ctx, x, y, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function applyShape(shape) {
    if (shape === currentShape) return;
    currentShape = shape;
    const LABELS = {
        square:    '◼  Cuadrado',
        rectangle: '▬  Rectángulo',
        triangle:  '▲  Triángulo',
        circle:    '●  Círculo',
        house:     '🏠  Casa',
        random:    '¡Pellizca para dibujar!'
    };
    shapeText.innerText = LABELS[shape] || shape;
    if (shape !== 'random') {
        particles.scale.set(1.5, 1.5, 1.5); // explosion pop
        playSuccessSound();
    }
    updateTargets(shape);
}

// ============================================================
//  THREE.JS ANIMATION LOOP
// ============================================================
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    const posArr = particles.geometry.attributes.position.array;
    const colArr = particles.geometry.attributes.color.array;
    const time   = clock.getElapsedTime();

    // Spring scale back to 1
    particles.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);

    let cX = 0, cY = 0;
    if (handState === 'open' && currentShape !== 'random') {
        cX = palmPos.x;
        cY = palmPos.y + 3.0;
    }

    for (let i = 0; i < NUM_PARTICLES; i++) {
        const px = posArr[i*3],   py = posArr[i*3+1], pz = posArr[i*3+2];
        let   tx = targets[i*3],  ty = targets[i*3+1], tz = targets[i*3+2];

        if (handState === 'closed' && currentShape !== 'random') {
            // Absorb: crush into palm
            tx = palmPos.x + (Math.random()-0.5) * 1.0;
            ty = palmPos.y + (Math.random()-0.5) * 1.0;
            tz = (Math.random()-0.5) * 1.0;
        } else if (handState === 'open' && currentShape !== 'random') {
            // Pedestal: float above palm
            tx += cX;
            ty += cY;
        }

        const speed = (handState === 'closed') ? 0.2 : 0.08;
        let nx = px + (tx - px) * speed;
        let ny = py + (ty - py) * speed;
        let nz = pz + (tz - pz) * speed;

        // Repulsion force field at finger tip (while pinching)
        if (isDrawing) {
            const dx = nx - fingerPos.x;
            const dy = ny - fingerPos.y;
            const d  = Math.sqrt(dx*dx + dy*dy);
            const R  = 3.0, F = 0.4;
            if (d > 0 && d < R) {
                const push = (R - d) * F;
                nx += (dx / d) * push;
                ny += (dy / d) * push;
                nz += (Math.random() - 0.5) * push;
            }
        }

        // Float / breathe
        if (handState !== 'closed') {
            ny += Math.sin(time * 2.0 + i * 0.1) * 0.01;
            nx += Math.cos(time * 1.5 + i * 0.1) * 0.01;
        }

        posArr[i*3]   = nx;
        posArr[i*3+1] = ny;
        posArr[i*3+2] = nz;

        // Lerp colors
        colArr[i*3]   += (targetColors[i*3]   - colArr[i*3])   * 0.05;
        colArr[i*3+1] += (targetColors[i*3+1] - colArr[i*3+1]) * 0.05;
        colArr[i*3+2] += (targetColors[i*3+2] - colArr[i*3+2]) * 0.05;
    }

    particles.geometry.attributes.position.needsUpdate = true;
    particles.geometry.attributes.color.needsUpdate    = true;

    // Slow rotation when not absorbing
    if (handState !== 'closed') {
        particles.rotation.y = Math.sin(time * 0.2) * 0.1;
        particles.rotation.x = Math.cos(time * 0.1) * 0.05;
    } else {
        particles.rotation.y += 0.05;
    }

    renderer.render(scene, threeCamera);
}

// Window resize
window.addEventListener('resize', () => {
    threeCamera.aspect = window.innerWidth / window.innerHeight;
    threeCamera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================
//  BOOT
// ============================================================
animate();   // Three.js runs immediately (particles visible during loading)
initApp();   // async: loads MediaPipe → camera → starts detectionLoop
