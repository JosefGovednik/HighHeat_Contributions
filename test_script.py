from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "model/weights2.pt"
VIDEO_PATH = "Test_videos/20250923_184300.mp4"

CONF = 0.05
IMG_SIZE = 1280
TARGET_FRAME = 196

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)


# -----------------------------
# UTILITIES
# -----------------------------

def run_inference(img, label):
    results = model(img, imgsz=IMG_SIZE, conf=CONF)[0]

    balls = []
    rubbers = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        if name == "ball":
            balls.append((conf, x1, y1, x2, y2))
        elif name == "rubber":
            rubbers.append((conf, x1, y1, x2, y2))

    print(f"\n[{label}]")
    print(f"balls: {balls}")
    print(f"rubbers: {rubbers}")

    return results


def tile_image(img, tile_size=640):
    h, w = img.shape[:2]
    tiles = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] < 100 or tile.shape[1] < 100:
                continue
            tiles.append(tile)

    return tiles


# -----------------------------
# MAIN LOOP (single frame test)
# -----------------------------

cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME - 1)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Could not read frame {TARGET_FRAME}")

print(f"Testing frame {TARGET_FRAME}")
print("Original frame shape:", frame.shape)


# =========================================================
# 1. FULL FRAME BASELINE
# =========================================================
run_inference(frame, "FULL FRAME")


# =========================================================
# 2. CROP AROUND DETECTION (if any rubber exists)
# =========================================================
results = model(frame, imgsz=IMG_SIZE, conf=CONF)[0]

crop_done = False

for box in results.boxes:
    cls = int(box.cls[0])
    name = model.names[cls]

    if name == "rubber":
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        r = 200  # crop radius

        crop = frame[
            max(0, cy-r):cy+r,
            max(0, cx-r):cx+r
        ]

        run_inference(crop, "CROP AROUND RUBBER")
        crop_done = True
        break

if not crop_done:
    print("\n[CROP TEST] No rubber found, skipping crop test")


# =========================================================
# 3. TILED INFERENCE TEST
# =========================================================
tiles = tile_image(frame, tile_size=640)

print(f"\n[TILES] Running inference on {len(tiles)} tiles")

for i, tile in enumerate(tiles):
    run_inference(tile, f"TILE {i}")