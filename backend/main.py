import asyncio
import websockets  # type: ignore
import cv2
import mediapipe as mp  # type: ignore
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import json
import time
import threading
import os
import urllib.request
import base64

# Global state for WebSocket broadcasting
state_lock = threading.Lock()
current_shape = "random"
finger_x, finger_y = 0.5, 0.5
palm_x, palm_y = 0.5, 0.5
is_drawing = False
hand_state = "other"  # 'pinching', 'open', 'closed', 'other'
current_frame_base64 = ""

history_shapes = []
last_shape_time = 0

clients = set()

# Download model if it does not exist
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe (esto tomará unos segundos)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Modelo descargado.")

# MediaPipe initialization (New Tasks API)
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

async def broadcast_state():
    """Continously sends the global state to all connected WebSocket clients."""
    while True:
        with state_lock:
            msg = json.dumps({
                "type": "update",
                "finger_x": finger_x,
                "finger_y": finger_y,
                "palm_x": palm_x,
                "palm_y": palm_y,
                "hand_state": hand_state,
                "shape": current_shape,
                "is_drawing": is_drawing,
                "frame": current_frame_base64
            })
            
        if clients:
            # Broadcast message
            await asyncio.gather(*[client.send(msg) for client in clients], return_exceptions=True)
            
        await asyncio.sleep(0.03) # Approx 30fps update rate

async def ws_handler(websocket, path="/"):
    """Handles new WebSocket connections."""
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_shape(points, img_w, img_h):
    """Detects geometric shape from drawn points using OpenCV."""
    if len(points) < 10:
        return None
    
    pts = np.array(points, dtype=np.int32)
    # Simplify polygon
    epsilon = 0.04 * cv2.arcLength(pts, True)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    
    vertices = len(approx)
    
    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        # Check aspect ratio to distinguish square from rectangle
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h
        if 0.5 <= aspect_ratio <= 1.5:
            return "square"
        else:
            return "rectangle"
    elif vertices > 6:
        return "circle"
    
    return None

def cv_loop():
    """Main computer vision loop (runs in a separate thread)."""
    global finger_x, finger_y, palm_x, palm_y, is_drawing, hand_state, current_shape, history_shapes, last_shape_time, current_frame_base64
    
    cap = cv2.VideoCapture(0)
    drawing_points = []
    
    HAND_CONNECTIONS = [
        (0,1), (1,2), (2,3), (3,4),       # Thumb
        (5,6), (6,7), (7,8),              # Index
        (9,10), (10,11), (11,12),         # Middle
        (13,14), (14,15), (15,16),        # Ring
        (17,18), (18,19), (19,20),        # Pinky
        (0,5), (5,9), (9,13), (13,17), (0,17) # Palm
    ]
    
    print("Iniciando cámara en modo 'Silencioso' (sin ventana OpenCV)...")
    print("Todo se verá en la página web.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        # Flip image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        img_h, img_w, _ = image.shape
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mp_image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect hands
        detection_result = detector.detect(mp_image)
        
        currently_pinching = False
        current_hand_state = "other"
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Dibujar esqueleto de la mano
                for connection in HAND_CONNECTIONS:
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
                    x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
                    cv2.line(image, (x1, y1), (x2, y2), (200, 200, 200), 2)
                    
                for i, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * img_w), int(lm.y * img_h)
                    cv2.circle(image, (cx, cy), 4, (255, 0, 0), -1)
                
                wrist = hand_landmarks[0]
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_tip = hand_landmarks[12]
                ring_tip = hand_landmarks[16]
                pinky_tip = hand_landmarks[20]
                
                palm_base = hand_landmarks[9] # Middle finger MCP (center of palm approx)
                
                with state_lock:
                    # Guardamos el centro del pinch o del dedo indice para el trazo
                    finger_x = (index_tip.x + thumb_tip.x) / 2.0
                    finger_y = (index_tip.y + thumb_tip.y) / 2.0
                    palm_x = palm_base.x
                    palm_y = palm_base.y
                
                pinch_dist = get_distance(thumb_tip, index_tip)
                
                # Calculate average distance of tips to wrist to determine open/closed
                dist_index = get_distance(index_tip, wrist)
                dist_middle = get_distance(middle_tip, wrist)
                dist_ring = get_distance(ring_tip, wrist)
                dist_pinky = get_distance(pinky_tip, wrist)
                avg_dist = (dist_index + dist_middle + dist_ring + dist_pinky) / 4.0
                
                # Coordenadas exactas para dibujar
                ix, iy = int(index_tip.x * img_w), int(index_tip.y * img_h)
                tx, ty = int(thumb_tip.x * img_w), int(thumb_tip.y * img_h)
                mid_x, mid_y = int((ix + tx) / 2), int((iy + ty) / 2)
                
                if pinch_dist < 0.05:
                    currently_pinching = True
                    current_hand_state = "pinching"
                    drawing_points.append([mid_x, mid_y])
                    # Resaltar ambos dedos y el centro
                    cv2.circle(image, (ix, iy), 10, (0, 255, 0), -1)
                    cv2.circle(image, (tx, ty), 10, (0, 255, 0), -1)
                    cv2.line(image, (ix, iy), (tx, ty), (0, 255, 0), 4)
                else:
                    cv2.circle(image, (ix, iy), 8, (0, 0, 255), -1)
                    cv2.circle(image, (tx, ty), 8, (0, 0, 255), -1)
                    if avg_dist < 0.2:
                        current_hand_state = "closed"
                    elif avg_dist >= 0.2:
                        current_hand_state = "open"
                    
        if is_drawing and not currently_pinching:
            # Gesture ended (pinch released), run shape detection
            if len(drawing_points) > 15:
                shape = detect_shape(drawing_points, img_w, img_h)
                if shape:
                    current_time = time.time()
                    
                    with state_lock:
                        if shape == "triangle" and "square" in history_shapes and (current_time - last_shape_time) < 5.0:
                            current_shape = "house"
                            history_shapes = []
                        else:
                            current_shape = shape
                            history_shapes.append(shape)
                            last_shape_time = current_time
                            
                        if len(history_shapes) > 5:
                            history_shapes.pop(0)
                        
            drawing_points = []
            
        with state_lock:
            is_drawing = currently_pinching
            hand_state = current_hand_state
            curr_shape_draw = current_shape
        
        # Draw the continuous line
        if len(drawing_points) > 1:
            for i in range(1, len(drawing_points)):
                cv2.line(image, tuple(drawing_points[i-1]), tuple(drawing_points[i]), (255, 255, 0), 3)
                
        # Overlay status text
        cv2.putText(image, f"Shape: {curr_shape_draw}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"State: {current_hand_state}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        # Encode image to base64 for the frontend
        small_image = cv2.resize(image, (640, 480))
        _, buffer = cv2.imencode('.jpg', small_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        with state_lock:
            current_frame_base64 = frame_base64
            
    cap.release()
    cv2.destroyAllWindows()
    import os
    os._exit(0)

async def main():
    # Start the computer vision loop in a separate thread
    t = threading.Thread(target=cv_loop, daemon=True)
    t.start()
    
    # Start the WebSocket server
    print("Iniciando servidor WebSocket en ws://localhost:8765...")
    server = await websockets.serve(ws_handler, "localhost", 8765)
    await broadcast_state()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Cerrando servidor.")
