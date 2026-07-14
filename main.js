// ============================================================
// GESTURA - Pure Browser Edition (MediaPipe Tasks Vision JS)
// No backend needed. Everything runs locally in your browser.
// ============================================================

import { HandLandmarker, FilesetResolver } from
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- DOM refs ----
const shapeText       = document.getElementById('shape-text');
const handStateBadge  = document.getElementById('hand-state-badge');
const loadingOverlay  = document.getElementById('loading-overlay');
const loaderFill      = document.getElementById('loader-fill');
const loaderError     = document.getElementById('loader-error');
const lmCanvas        = document.getElementById('landmark-canvas');
const lmCtx           = lmCanvas.getContext('2d');
const videoEl         = document.getElementById('input-video');

// ---- Hand skeleton connections ----
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [5,6],[6,7],[7,8],
    [9,10],[10,11],[11,12],
    [13,14],[14,15],[15,16],
    [17,18],[18,19],[19,20],
    [0,5],[5,9],[9,13],[13,17],[0,17]
];

// ---- Global gesture state ----
let fingerPos   = new THREE.Vector2(0, 0);
let palmPos     = new THREE.Vector2(0, 0);
let isDrawing   = false;
let handState   = "other";
let currentShape = "random";
let historyShapes = [];
let lastShapeTime = 0;
let drawingPoints = []; // pixel-space coords on the landmark canvas

// ============================================================
// THREE.JS SCENE SETUP
// ============================================================
const numParticles = 10000;
const particleSize = 0.08;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050505, 0.02);

const threeCamera = new THREE.PerspectiveCamera(
    75, window.innerWidth / window.innerHeight, 0.1, 1000
);
threeCamera.position.z = 8;

const renderer = new THREE.WebGLRenderer({
    antialias: true, alpha: true, powerPreference: "high-performance"
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// ---- Particle buffers ----
const geometry    = new THREE.BufferGeometry();
const positions   = new Float32Array(numParticles * 3);
const targets     = new Float32Array(numParticles * 3);
const colors      = new Float32Array(numParticles * 3);
const targetColors = new Float32Array(numParticles * 3);

// ---- Color palettes ----
const palettes = {
    random:    [0x00f2fe, 0x4facfe, 0xff0844, 0xffb199, 0xf5576c],
    square:    [0x00c6ff, 0x0072ff, 0x4facfe, 0x00f2fe],
    triangle:  [0x00b09b, 0x96c93d, 0x0ba360, 0x3cba92],
    circle:    [0xb224ef, 0x7579ff, 0x667eea, 0x764ba2],
    rectangle: [0x43e97b, 0x38f9d7, 0x00cdac, 0x78ffd6],
    house:     [0xf6d365, 0xfda085, 0xffd194, 0x70e1f5]
};

function getRandomColor(paletteName) {
    const pal = palettes[paletteName] || palettes.random;
    return new THREE.Color(pal[Math.floor(Math.random() * pal.length)]);
}

// ---- Initialize particle positions ----
for (let i = 0; i < numParticles; i++) {
    const x = (Math.random() - 0.5) * 20;
    const y = (Math.random() - 0.5) * 20;
    const z = (Math.random() - 0.5) * 10;
    positions[i*3] = x; positions[i*3+1] = y; positions[i*3+2] = z;
    targets[i*3]   = x; targets[i*3+1]   = y; targets[i*3+2]   = z;
    const c = getRandomColor('random');
    colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
    targetColors[i*3] = c.r; targetColors[i*3+1] = c.g; targetColors[i*3+2] = c.b;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color',    new THREE.BufferAttribute(colors,    3));

// ---- Soft glow particle texture ----
const ptCanvas = document.createElement('canvas');
ptCanvas.width = ptCanvas.height = 32;
const ptCtx = ptCanvas.getContext('2d');
const grad = ptCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
grad.addColorStop(0, 'rgba(255,255,255,1)');
grad.addColorStop(1, 'rgba(255,255,255,0)');
ptCtx.fillStyle = grad;
ptCtx.fillRect(0, 0, 32, 32);
const particleTexture = new THREE.CanvasTexture(ptCanvas);

const material = new THREE.PointsMaterial({
    size: particleSize,
    vertexColors: true,
    map: particleTexture,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    transparent: true,
    opacity: 0.8
});

const particles = new THREE.Points(geometry, material);
scene.add(particles);

// ============================================================
// SHAPE GENERATION FUNCTIONS (same as Python backend)
// ============================================================
function getSquarePoint() {
    const size = 5, edge = Math.floor(Math.random() * 4), t = Math.random() * size - size / 2;
    if (edge === 0) return {x: t, y:  size/2};
    if (edge === 1) return {x: t, y: -size/2};
    if (edge === 2) return {x:  size/2, y: t};
    return {x: -size/2, y: t};
}

function getRectanglePoint() {
    const W = 7, H = 4, edge = Math.floor(Math.random() * 4);
    if (edge === 0) return {x: Math.random()*W - W/2, y:  H/2};
    if (edge === 1) return {x: Math.random()*W - W/2, y: -H/2};
    if (edge === 2) return {x:  W/2, y: Math.random()*H - H/2};
    return {x: -W/2, y: Math.random()*H - H/2};
}

function getTrianglePoint() {
    const s = 5, edge = Math.floor(Math.random() * 3), t = Math.random();
    if (edge === 0) return {x: t*(-s/2) + (1-t)*0,    y: t*(-s/2) + (1-t)*(s/2)};
    if (edge === 1) return {x: t*(s/2)  + (1-t)*0,    y: t*(-s/2) + (1-t)*(s/2)};
    return {x: t*(s/2) + (1-t)*(-s/2), y: -s/2};
}

function getCirclePoint() {
    const angle = Math.random() * Math.PI * 2;
    const r = 2.5 + (Math.random() - 0.5) * 0.5;
    return {x: Math.cos(angle) * r, y: Math.sin(angle) * r};
}

function getHousePoint() {
    const size = 4;
    if (Math.random() < 0.6) {
        const edge = Math.floor(Math.random() * 3);
        const t = Math.random() * size - size/2;
        if (edge === 0) return {x: t,       y: -size/2};
        if (edge === 1) return {x: -size/2, y: t};
        return {x: size/2, y: t};
    }
    const t = Math.random(), edge = Math.floor(Math.random() * 2);
    if (edge === 0) return {x: t*(-size/2 - 0.5) + (1-t)*0, y: t*(size/2) + (1-t)*(size/2+2)};
    return {x: t*(size/2 + 0.5) + (1-t)*0, y: t*(size/2) + (1-t)*(size/2+2)};
}

function updateTargets(shape) {
    for (let i = 0; i < numParticles; i++) {
        let p = {x: (Math.random()-0.5)*20, y: (Math.random()-0.5)*20};
        if (shape === 'square')    p = getSquarePoint();
        else if (shape === 'rectangle') p = getRectanglePoint();
        else if (shape === 'triangle')  p = getTrianglePoint();
        else if (shape === 'house')     p = getHousePoint();
        else if (shape === 'circle')    p = getCirclePoint();

        const noise = (shape === 'random') ? 0 : (Math.random()-0.5)*0.4;
        targets[i*3]   = p.x + noise;
        targets[i*3+1] = p.y + noise;
        targets[i*3+2] = (Math.random()-0.5)*1.0;

        const c = getRandomColor(shape);
        targetColors[i*3]   = c.r;
        targetColors[i*3+1] = c.g;
        targetColors[i*3+2] = c.b;
    }
}

// ============================================================
// AUDIO
// ============================================================
let audioCtx = null;

function initAudio() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
}

function playSuccessSound() {
    initAudio();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(440, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.1);
    osc.frequency.exponentialRampToValueAtTime(1108, audioCtx.currentTime + 0.2);
    gain.gain.setValueAtTime(0, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0.3, audioCtx.currentTime + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.6);
    osc.start(audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.6);
}

function playAbsorbSound() {
    initAudio();
    const osc = audioCtx.createOscillator();
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
// SHAPE DETECTION (Douglas–Peucker, equivalent to cv2.approxPolyDP)
// ============================================================
function perpDist(pt, a, b) {
    const dx = b[0]-a[0], dy = b[1]-a[1];
    const len = Math.sqrt(dx*dx + dy*dy);
    if (len === 0) return Math.sqrt((pt[0]-a[0])**2 + (pt[1]-a[1])**2);
    return Math.abs(dy*pt[0] - dx*pt[1] + b[0]*a[1] - b[1]*a[0]) / len;
}

function rdp(pts, eps) {
    if (pts.length < 3) return pts;
    let maxD = 0, maxI = 0;
    for (let i = 1; i < pts.length-1; i++) {
        const d = perpDist(pts[i], pts[0], pts[pts.length-1]);
        if (d > maxD) { maxD = d; maxI = i; }
    }
    if (maxD > eps) {
        const l = rdp(pts.slice(0, maxI+1), eps);
        const r = rdp(pts.slice(maxI), eps);
        return [...l.slice(0, -1), ...r];
    }
    return [pts[0], pts[pts.length-1]];
}

function arcLength(pts) {
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
        const dx = pts[i][0]-pts[i-1][0], dy = pts[i][1]-pts[i-1][1];
        total += Math.sqrt(dx*dx + dy*dy);
    }
    // close
    const dx = pts[0][0]-pts[pts.length-1][0], dy = pts[0][1]-pts[pts.length-1][1];
    return total + Math.sqrt(dx*dx + dy*dy);
}

function detectShape(pts) {
    if (pts.length < 10) return null;
    const perim = arcLength(pts);
    const eps = 0.04 * perim;
    const closed = [...pts, pts[0]];
    const approx = rdp(closed, eps);
    const verts = approx.length - 1;

    if (verts === 3) return 'triangle';
    if (verts === 4) {
        const xs = approx.map(p => p[0]);
        const ys = approx.map(p => p[1]);
        const w = Math.max(...xs) - Math.min(...xs);
        const h = Math.max(...ys) - Math.min(...ys);
        const ar = w / h;
        return (ar >= 0.5 && ar <= 1.5) ? 'square' : 'rectangle';
    }
    if (verts > 6) return 'circle';
    return null;
}

// ============================================================
// MEDIAPIPE + CAMERA SETUP
// ============================================================
let handLandmarker = null;
let lastVideoTime = -1;

async function initApp() {
    // Step 1: Load MediaPipe
    setLoaderProgress(20, "Cargando MediaPipe...");
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        setLoaderProgress(55, "Cargando modelo de mano...");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1
        });
    } catch (e) {
        showError("Error cargando MediaPipe: " + e.message);
        return;
    }

    // Step 2: Request camera
    setLoaderProgress(80, "Solicitando permiso de cámara...");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" }
        });
        videoEl.srcObject = stream;
        await new Promise(res => videoEl.onloadeddata = res);
    } catch (e) {
        showError("No se pudo acceder a la cámara: " + e.message);
        return;
    }

    setLoaderProgress(100, "¡Listo!");
    setTimeout(() => {
        loadingOverlay.classList.add('fade-out');
        setTimeout(() => loadingOverlay.style.display = 'none', 650);
    }, 400);

    shapeText.innerText = "¡Dibuja en el aire!";
    requestAnimationFrame(detectionLoop);
}

function setLoaderProgress(pct, msg) {
    loaderFill.style.width = pct + '%';
    document.querySelector('.loader-subtitle').textContent = msg;
}

function showError(msg) {
    loaderError.style.display = 'block';
    loaderError.textContent = msg;
}

// ============================================================
// HAND DETECTION LOOP
// ============================================================
function dist2D(a, b) {
    return Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2);
}

function detectionLoop(now) {
    requestAnimationFrame(detectionLoop);

    if (!handLandmarker || videoEl.readyState < 2) return;
    if (videoEl.currentTime === lastVideoTime) return;
    lastVideoTime = videoEl.currentTime;

    // Resize landmark canvas to match window
    if (lmCanvas.width !== window.innerWidth || lmCanvas.height !== window.innerHeight) {
        lmCanvas.width  = window.innerWidth;
        lmCanvas.height = window.innerHeight;
    }

    // Run detection
    const result = handLandmarker.detectForVideo(videoEl, now);

    // Clear canvas, draw camera frame
    lmCtx.clearRect(0, 0, lmCanvas.width, lmCanvas.height);
    lmCtx.drawImage(videoEl, 0, 0, lmCanvas.width, lmCanvas.height);

    let currentlyPinching = false;
    let currentHandState  = "other";

    if (result.landmarks && result.landmarks.length > 0) {
        const lm = result.landmarks[0]; // First hand only

        const W = lmCanvas.width, H = lmCanvas.height;

        // Draw skeleton
        lmCtx.strokeStyle = 'rgba(200,200,200,0.7)';
        lmCtx.lineWidth = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
            lmCtx.beginPath();
            lmCtx.moveTo(lm[a].x * W, lm[a].y * H);
            lmCtx.lineTo(lm[b].x * W, lm[b].y * H);
            lmCtx.stroke();
        }

        // Key landmarks
        const wrist      = lm[0];
        const thumbTip   = lm[4];
        const indexTip   = lm[8];
        const middleTip  = lm[12];
        const ringTip    = lm[16];
        const pinkyTip   = lm[20];
        const palmBase   = lm[9];

        // Update global positions for Three.js
        fingerPos.x = ((indexTip.x + thumbTip.x) / 2 - 0.5) * 20;
        fingerPos.y = -((indexTip.y + thumbTip.y) / 2 - 0.5) * 15;
        palmPos.x   = (palmBase.x - 0.5) * 20;
        palmPos.y   = -(palmBase.y - 0.5) * 15;

        const pinchDist = dist2D(thumbTip, indexTip);

        // Average fingertip distance to wrist for open/closed
        const avgDist = (
            dist2D(indexTip, wrist) + dist2D(middleTip, wrist) +
            dist2D(ringTip, wrist)  + dist2D(pinkyTip, wrist)
        ) / 4;

        const ix = indexTip.x * W, iy = indexTip.y * H;
        const tx = thumbTip.x  * W, ty = thumbTip.y  * H;
        const midX = (ix + tx) / 2, midY = (iy + ty) / 2;

        if (pinchDist < 0.05) {
            currentlyPinching = true;
            currentHandState  = "pinching";
            drawingPoints.push([midX, midY]);

            // Draw pinch highlight
            lmCtx.strokeStyle = '#00ff88';
            lmCtx.lineWidth = 4;
            lmCtx.beginPath(); lmCtx.moveTo(ix, iy); lmCtx.lineTo(tx, ty); lmCtx.stroke();
            drawDot(lmCtx, ix, iy, 10, '#00ff88');
            drawDot(lmCtx, tx, ty, 10, '#00ff88');
        } else {
            drawDot(lmCtx, ix, iy, 8, '#ff4444');
            drawDot(lmCtx, tx, ty, 8, '#ff4444');
            currentHandState = (avgDist < 0.2) ? "closed" : "open";
        }

        // Draw all landmarks
        for (const lmk of lm) {
            drawDot(lmCtx, lmk.x * W, lmk.y * H, 4, 'rgba(100,180,255,0.9)');
        }
    }

    // ---- Gesture state transitions ----
    if (isDrawing && !currentlyPinching) {
        // Pinch released – run shape detection
        if (drawingPoints.length > 15) {
            const shape = detectShape(drawingPoints);
            if (shape) {
                const now = performance.now() / 1000;
                if (shape === 'triangle' && historyShapes.includes('square') && (now - lastShapeTime) < 5.0) {
                    applyShape('house');
                    historyShapes = [];
                } else {
                    applyShape(shape);
                    historyShapes.push(shape);
                    lastShapeTime = now;
                    if (historyShapes.length > 5) historyShapes.shift();
                }
            }
        }
        drawingPoints = [];
    }

    // Draw the trail
    if (drawingPoints.length > 1) {
        lmCtx.strokeStyle = 'rgba(255, 255, 0, 0.85)';
        lmCtx.lineWidth = 3;
        lmCtx.beginPath();
        lmCtx.moveTo(drawingPoints[0][0], drawingPoints[0][1]);
        for (let i = 1; i < drawingPoints.length; i++) {
            lmCtx.lineTo(drawingPoints[i][0], drawingPoints[i][1]);
        }
        lmCtx.stroke();
    }

    // Audio trigger on absorb
    if (currentHandState === "closed" && handState !== "closed" && currentShape !== "random") {
        playAbsorbSound();
    }
    if (currentlyPinching) initAudio();

    isDrawing = currentlyPinching;
    handState = currentHandState;

    // Update badge
    const stateLabels = {
        pinching: "✍️  Dibujando",
        open:     "🖐  Mano abierta",
        closed:   "✊  Puño cerrado",
        other:    "—"
    };
    handStateBadge.textContent = stateLabels[handState] || "—";
}

function drawDot(ctx, x, y, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function applyShape(shape) {
    if (shape === currentShape) return;
    currentShape = shape;
    if (shape !== 'random') {
        const labels = {
            square:    'Cuadrado ◼',
            rectangle: 'Rectángulo ▬',
            triangle:  'Triángulo ▲',
            circle:    'Círculo ●',
            house:     '🏠 Casa'
        };
        shapeText.innerText = labels[shape] || shape;
        particles.scale.set(1.4, 1.4, 1.4);
        playSuccessSound();
    } else {
        shapeText.innerText = "¡Dibuja en el aire!";
    }
    updateTargets(shape);
}

// ============================================================
// THREE.JS ANIMATION LOOP
// ============================================================
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    const posArr  = particles.geometry.attributes.position.array;
    const colArr  = particles.geometry.attributes.color.array;
    const time    = clock.getElapsedTime();

    // Spring scale back
    particles.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);

    let centerX = 0, centerY = 0;
    if (handState === "open" && currentShape !== "random") {
        centerX = palmPos.x;
        centerY = palmPos.y + 3.0;
    }

    for (let i = 0; i < numParticles; i++) {
        const px = posArr[i*3], py = posArr[i*3+1], pz = posArr[i*3+2];
        let tx = targets[i*3], ty = targets[i*3+1], tz = targets[i*3+2];

        if (handState === "closed" && currentShape !== "random") {
            tx = palmPos.x + (Math.random()-0.5)*1.0;
            ty = palmPos.y + (Math.random()-0.5)*1.0;
            tz = (Math.random()-0.5)*1.0;
        } else if (handState === "open" && currentShape !== "random") {
            tx += centerX;
            ty += centerY;
        }

        const speed = (handState === "closed") ? 0.2 : 0.08;
        let nx = px + (tx-px)*speed;
        let ny = py + (ty-py)*speed;
        let nz = pz + (tz-pz)*speed;

        // Finger force field (only while drawing/pinching)
        if (isDrawing) {
            const dx = nx - fingerPos.x;
            const dy = ny - fingerPos.y;
            const d  = Math.sqrt(dx*dx + dy*dy);
            const R  = 3.0, F = 0.4;
            if (d < R) {
                const push = (R-d)*F;
                nx += (dx/d)*push;
                ny += (dy/d)*push;
                nz += (Math.random()-0.5)*push;
            }
        }

        // Float effect
        if (handState !== "closed") {
            ny += Math.sin(time*2 + i*0.1)*0.01;
            nx += Math.cos(time*1.5 + i*0.1)*0.01;
        }

        posArr[i*3] = nx; posArr[i*3+1] = ny; posArr[i*3+2] = nz;

        // Lerp colors
        colArr[i*3]   += (targetColors[i*3]   - colArr[i*3])   * 0.05;
        colArr[i*3+1] += (targetColors[i*3+1] - colArr[i*3+1]) * 0.05;
        colArr[i*3+2] += (targetColors[i*3+2] - colArr[i*3+2]) * 0.05;
    }

    particles.geometry.attributes.position.needsUpdate = true;
    particles.geometry.attributes.color.needsUpdate    = true;

    // Rotation
    if (handState !== "closed") {
        particles.rotation.y = Math.sin(time*0.2)*0.1;
        particles.rotation.x = Math.cos(time*0.1)*0.05;
    } else {
        particles.rotation.y += 0.05;
    }

    renderer.render(scene, threeCamera);
}

// ---- Resize ----
window.addEventListener('resize', () => {
    threeCamera.aspect = window.innerWidth / window.innerHeight;
    threeCamera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================
// BOOT
// ============================================================
animate();
initApp();
