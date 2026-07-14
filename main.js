// Gestura - Magic Particles & Interactive Audio
const numParticles = 10000;
const particleSize = 0.08;

// Scene setup
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050505, 0.02);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 8;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// Particle Geometry
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(numParticles * 3);
const targets = new Float32Array(numParticles * 3);
const colors = new Float32Array(numParticles * 3);
const targetColors = new Float32Array(numParticles * 3);

// Vibrant neon color palettes
const palettes = {
    random: [0x00f2fe, 0x4facfe, 0xff0844, 0xffb199, 0xf5576c], // Mix
    square: [0x00c6ff, 0x0072ff, 0x4facfe, 0x00f2fe],           // Blues
    triangle: [0x00b09b, 0x96c93d, 0x0ba360, 0x3cba92],         // Greens
    circle: [0xb224ef, 0x7579ff, 0x667eea, 0x764ba2],           // Purples
    house: [0xf6d365, 0xfda085, 0xffd194, 0x70e1f5]             // Golds
};

function getRandomColor(paletteName) {
    const pal = palettes[paletteName] || palettes.random;
    const c = new THREE.Color(pal[Math.floor(Math.random() * pal.length)]);
    return c;
}

// Initialize random positions
for (let i = 0; i < numParticles; i++) {
    const x = (Math.random() - 0.5) * 20;
    const y = (Math.random() - 0.5) * 20;
    const z = (Math.random() - 0.5) * 10;
    
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    
    targets[i * 3] = x;
    targets[i * 3 + 1] = y;
    targets[i * 3 + 2] = z;
    
    const color = getRandomColor('random');
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
    
    targetColors[i * 3] = color.r;
    targetColors[i * 3 + 1] = color.g;
    targetColors[i * 3 + 2] = color.b;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

// Custom Material for soft glowing particles
const canvas = document.createElement('canvas');
canvas.width = 32;
canvas.height = 32;
const context = canvas.getContext('2d');
const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
gradient.addColorStop(0, 'rgba(255,255,255,1)');
gradient.addColorStop(1, 'rgba(255,255,255,0)');
context.fillStyle = gradient;
context.fillRect(0,0,32,32);
const particleTexture = new THREE.CanvasTexture(canvas);

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

// ----- SHAPE GENERATION -----
function getSquarePoint() {
    const size = 5;
    const edge = Math.floor(Math.random() * 4);
    const t = Math.random() * size - size/2;
    if (edge === 0) return {x: t, y: size/2};
    if (edge === 1) return {x: t, y: -size/2};
    if (edge === 2) return {x: size/2, y: t};
    return {x: -size/2, y: t};
}

function getRectanglePoint() {
    const width = 7;
    const height = 4;
    const edge = Math.floor(Math.random() * 4);
    if (edge === 0) return {x: Math.random() * width - width/2, y: height/2};
    if (edge === 1) return {x: Math.random() * width - width/2, y: -height/2};
    if (edge === 2) return {x: width/2, y: Math.random() * height - height/2};
    return {x: -width/2, y: Math.random() * height - height/2};
}

function getTrianglePoint() {
    const size = 5;
    const edge = Math.floor(Math.random() * 3);
    const t = Math.random();
    if (edge === 0) return { x: t * (-size/2) + (1-t) * 0, y: t * (-size/2) + (1-t) * (size/2) };
    if (edge === 1) return { x: t * (size/2) + (1-t) * 0, y: t * (-size/2) + (1-t) * (size/2) };
    return { x: t * (size/2) + (1-t) * (-size/2), y: -size/2 };
}

function getCirclePoint() {
    const angle = Math.random() * Math.PI * 2;
    const r = 2.5 + (Math.random() - 0.5) * 0.5;
    return { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
}

function getHousePoint() {
    const size = 4;
    const part = Math.random();
    if (part < 0.6) {
        // Walls
        const edge = Math.floor(Math.random() * 3);
        const t = Math.random() * size - size/2;
        if (edge === 0) return {x: t, y: -size/2}; 
        if (edge === 1) return {x: -size/2, y: t}; 
        return {x: size/2, y: t}; 
    } else {
        // Roof
        const t = Math.random();
        const edge = Math.floor(Math.random() * 2);
        if (edge === 0) return { x: t * (-size/2 - 0.5) + (1-t) * 0, y: t * (size/2) + (1-t) * (size/2 + 2) };
        return { x: t * (size/2 + 0.5) + (1-t) * 0, y: t * (size/2) + (1-t) * (size/2 + 2) };
    }
}

function updateTargets(shape) {
    for (let i = 0; i < numParticles; i++) {
        let point = { x: (Math.random() - 0.5) * 20, y: (Math.random() - 0.5) * 20 };
        
        if (shape === 'square') point = getSquarePoint();
        else if (shape === 'rectangle') point = getRectanglePoint();
        else if (shape === 'triangle') point = getTrianglePoint();
        else if (shape === 'house') point = getHousePoint();
        else if (shape === 'circle') point = getCirclePoint();
        
        const noise = (shape === 'random') ? 0 : (Math.random() - 0.5) * 0.4;
        
        targets[i * 3] = point.x + noise;
        targets[i * 3 + 1] = point.y + noise;
        targets[i * 3 + 2] = (Math.random() - 0.5) * 1.0;
        
        // Update Target Colors
        const color = getRandomColor(shape);
        targetColors[i * 3] = color.r;
        targetColors[i * 3 + 1] = color.g;
        targetColors[i * 3 + 2] = color.b;
    }
}

// ----- WEB AUDIO API -----
let audioCtx = null;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

function playSuccessSound() {
    initAudio();
    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    
    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    osc.type = 'sine';
    // Arpeggio / Chime effect
    osc.frequency.setValueAtTime(440, audioCtx.currentTime); 
    osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.1); 
    osc.frequency.exponentialRampToValueAtTime(1108, audioCtx.currentTime + 0.2); // C#6
    
    gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.3, audioCtx.currentTime + 0.05);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.6);
    
    osc.start(audioCtx.currentTime);
    osc.stop(audioCtx.currentTime + 0.6);
}

function playAbsorbSound() {
    initAudio();
    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    
    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(400, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(40, audioCtx.currentTime + 0.3); // Deep drop
    
    gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.4, audioCtx.currentTime + 0.05);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
    
    osc.start(audioCtx.currentTime);
    osc.stop(audioCtx.currentTime + 0.3);
}

// ----- WEBSOCKET CONNECTION -----
let ws;
const shapeText = document.getElementById('shape-text');
const bgVideo = document.getElementById('bg-video');

let fingerPos = new THREE.Vector2(0, 0);
let palmPos = new THREE.Vector2(0, 0);
let isDrawing = false;
let handState = "other";
let currentShape = 'random';

function connectWebSocket() {
    const wsUrl = 'ws://' + window.location.hostname + ':8765';
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        shapeText.innerText = "¡Dibuja en el aire!";
        console.log("WebSocket connected");
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Map [0, 1] camera coordinates to 3D space coordinates
        fingerPos.x = (data.finger_x - 0.5) * 20; 
        fingerPos.y = -(data.finger_y - 0.5) * 15;
        
        palmPos.x = (data.palm_x - 0.5) * 20;
        palmPos.y = -(data.palm_y - 0.5) * 15;
        
        if (data.is_drawing) {
            initAudio(); // Required to unlock audio on first gesture
        }
        
        isDrawing = data.is_drawing;
        
        // Track state change for audio
        if (data.hand_state === "closed" && handState !== "closed" && currentShape !== "random") {
            playAbsorbSound();
        }
        
        handState = data.hand_state || "other";
        
        if (data.shape !== currentShape) {
            currentShape = data.shape;
            if (currentShape !== "random") {
                shapeText.innerText = `Forma: ${currentShape}`;
                
                // Explosion effect
                particles.scale.set(1.4, 1.4, 1.4); 
                
                // Audio effect
                playSuccessSound();
            } else {
                shapeText.innerText = "¡Dibuja en el aire!";
            }
            updateTargets(currentShape);
        }
        
        // Update background camera feed if available
        if (data.frame) {
            bgVideo.src = "data:image/jpeg;base64," + data.frame;
        }
    };
    
    ws.onclose = () => {
        shapeText.innerText = "Servidor Desconectado. Reintentando...";
        setTimeout(connectWebSocket, 2000);
    };
    
    ws.onerror = (err) => {
        console.error("WebSocket Error", err);
    }
}

connectWebSocket();

// Click to unlock audio just in case
window.addEventListener('click', () => {
    initAudio();
});

// ----- ANIMATION LOOP -----
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    
    const positions = particles.geometry.attributes.position.array;
    const colorsArr = particles.geometry.attributes.color.array;
    const time = clock.getElapsedTime();
    
    // Scale spring back
    particles.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
    
    // Target Center logic based on hand state
    let targetCenterX = 0;
    let targetCenterY = 0;
    
    if (handState === "open" && currentShape !== "random") {
        // Pedestal mode: Center shape on the palm
        targetCenterX = palmPos.x;
        targetCenterY = palmPos.y + 3.0; // Float slightly above the palm
    }

    for (let i = 0; i < numParticles; i++) {
        const px = positions[i * 3];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        
        let tx = targets[i * 3];
        let ty = targets[i * 3 + 1];
        let tz = targets[i * 3 + 2];
        
        if (handState === "closed" && currentShape !== "random") {
            // Absorb mode: Crush into palm
            tx = palmPos.x + (Math.random() - 0.5) * 1.0;
            ty = palmPos.y + (Math.random() - 0.5) * 1.0;
            tz = (Math.random() - 0.5) * 1.0;
        } else if (handState === "open" && currentShape !== "random") {
            // Apply pedestal offset
            tx += targetCenterX;
            ty += targetCenterY;
        }
        
        // 1. Lerp to target positions
        let speed = (handState === "closed") ? 0.2 : 0.08; // Collapse fast
        let newX = px + (tx - px) * speed;
        let newY = py + (ty - py) * speed;
        let newZ = pz + (tz - pz) * speed;
        
        // 2. Force field (Finger) - Only apply if pinching/drawing
        if (isDrawing) {
            const dx = newX - fingerPos.x;
            const dy = newY - fingerPos.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            const interactionRadius = 3.0;
            const force = 0.4;
            
            if (dist < interactionRadius) {
                const push = (interactionRadius - dist) * force;
                newX += (dx / dist) * push;
                newY += (dy / dist) * push;
                newZ += (Math.random() - 0.5) * push;
            }
        }
        
        // 3. Float effect
        if (handState !== "closed") {
            newY += Math.sin(time * 2 + i * 0.1) * 0.01;
            newX += Math.cos(time * 1.5 + i * 0.1) * 0.01;
        }
        
        positions[i * 3] = newX;
        positions[i * 3 + 1] = newY;
        positions[i * 3 + 2] = newZ;
        
        // 4. Lerp Colors
        colorsArr[i * 3]     += (targetColors[i * 3]     - colorsArr[i * 3]) * 0.05;
        colorsArr[i * 3 + 1] += (targetColors[i * 3 + 1] - colorsArr[i * 3 + 1]) * 0.05;
        colorsArr[i * 3 + 2] += (targetColors[i * 3 + 2] - colorsArr[i * 3 + 2]) * 0.05;
    }
    
    particles.geometry.attributes.position.needsUpdate = true;
    particles.geometry.attributes.color.needsUpdate = true;
    
    // Rotate the whole system slowly if not collapsed
    if (handState !== "closed") {
        particles.rotation.y = Math.sin(time * 0.2) * 0.1;
        particles.rotation.x = Math.cos(time * 0.1) * 0.05;
    } else {
        // Spin fast while collapsed
        particles.rotation.y += 0.05;
    }
    
    renderer.render(scene, camera);
}

animate();

// Resize handling
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
