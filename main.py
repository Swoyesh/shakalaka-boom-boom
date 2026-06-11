import math
import os
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame
import torch

from train import QuickDNN, build_model
from utils.canvas import drawline
# from utils.generate import generate_image
from utils.gesture import is_pencil_grip, is_thumbs_up
from utils.overlay import draw_pencil
from utils.preprocess import preprocess_canvas


def save_canvas(canvas, label):
    if " " in label:
        label = label.replace(" ", "_")
    folder = f"collected_data/{label}"
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.endswith(".png")])
    num = existing + 1
    cv2.imwrite(f"{folder}/{label}_{num}.png", canvas)


def draw_thumb(frame, original_frame, hand_landmarks, w, h):
    thickness = 10
    pts = [
        (int(hand_landmarks[idx].x * w), int(hand_landmarks[idx].y * h))
        for idx in [1, 2, 3, 4]
    ]

    left_side = []
    right_side = []

    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            continue
        px, py = -dy / length, dx / length
        left_side.append((int(x1 + px * thickness), int(y1 + py * thickness)))
        right_side.append((int(x1 - px * thickness), int(y1 - py * thickness)))

    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length > 0:
        px, py = -dy / length, dx / length
        left_side.append((int(x2 + px * thickness), int(y2 + py * thickness)))
        right_side.append((int(x2 - px * thickness), int(y2 - py * thickness)))

    polygon = np.array(left_side + right_side[::-1], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    frame[mask == 255] = original_frame[mask == 255]


def overlay_image(base, img, x, y, new_w, new_h):
    img_resized = cv2.resize(img, (new_w, new_h))
    if img_resized.shape[2] == 4:
        alpha = img_resized[:, :, 3:4] / 255.0
        rgb = img_resized[:, :, :3]
    else:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        alpha = mask[:, :, np.newaxis] / 255.0
        rgb = img_resized[:, :, :3]
    x2, y2 = x + new_w, y + new_h
    x, y = max(x, 0), max(y, 0)
    roi = base[y:y2, x:x2]
    if roi.shape[:2] != rgb.shape[:2]:
        return
    base[y:y2, x:x2] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)


def draw_title_screen(frame, w, h):
    bg = cv2.resize(bg_img, (w, h))
    cv2.addWeighted(bg, 0.75, frame, 0.25, 0, frame)
    frame[:] = cv2.addWeighted(bg, 0.75, frame, 0.25, 0, None)

    logo_w, logo_h = 420, 160
    overlay_image(frame, logo_img, w // 2 - logo_w // 2, 30, logo_w, logo_h)

    text = "Which hand is your magic hand?"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    cv2.putText(
        frame,
        text,
        (w // 2 - tw // 2, h // 2 - 20),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (30, 30, 30),
        4,
    )
    cv2.putText(
        frame,
        text,
        (w // 2 - tw // 2, h // 2 - 20),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (255, 255, 255),
        2,
    )

    hand_w, hand_h = 180, 220

    # Right hand button
    rx = w // 2 - 320
    ry = h // 2 + 10
    cv2.rectangle(
        frame,
        (rx - 10, ry - 10),
        (rx + hand_w + 10, ry + hand_h + 10),
        (40, 40, 160),
        -1,
    )
    cv2.rectangle(
        frame,
        (rx - 10, ry - 10),
        (rx + hand_w + 10, ry + hand_h + 10),
        (255, 255, 255),
        2,
    )
    overlay_image(frame, left_hand_img, rx, ry, hand_w, hand_h)
    cv2.putText(
        frame,
        "L",
        (rx + hand_w - 28, ry + hand_h - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (30, 30, 30),
        4,
    )
    cv2.putText(
        frame,
        "L",
        (rx + hand_w - 28, ry + hand_h - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 220, 80),
        2,
    )

    lx = w // 2 + 130
    ly = h // 2 + 10
    cv2.rectangle(
        frame,
        (lx - 10, ly - 10),
        (lx + hand_w + 10, ly + hand_h + 10),
        (160, 60, 0),
        -1,
    )
    cv2.rectangle(
        frame,
        (lx - 10, ly - 10),
        (lx + hand_w + 10, ly + hand_h + 10),
        (255, 255, 255),
        2,
    )
    overlay_image(frame, right_hand_img, lx, ly, hand_w, hand_h)
    cv2.putText(
        frame,
        "R",
        (lx + hand_w - 28, ly + hand_h - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (30, 30, 30),
        4,
    )
    cv2.putText(
        frame,
        "R",
        (lx + hand_w - 28, ly + hand_h - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 220, 80),
        2,
    )

    cv2.imshow("Shakalaka", frame)


def draw_prediction_screen(
    combined,
    w,
    h,
    top3_labels,
    top3_probs,
    flash_start,
    typing_mode=False,
    typed_text="",
):
    dark = np.zeros_like(combined)
    cv2.addWeighted(dark, 0.5, combined, 0.5, 0, combined)

    elapsed = time.time() - flash_start
    if elapsed < 0.15:
        alpha = 1.0 - (elapsed / 0.15)
        combined[:] = np.clip(
            combined * (1 - alpha * 0.75) + 255 * (alpha * 0.75), 0, 255
        ).astype(np.uint8)

    s = w / 1280.0
    t2 = max(1, int(round(2 * s)))
    t1 = max(1, int(round(1 * s)))

    header = "WHAT DID YOU DRAW?"
    hs = 1.5 * s
    (tw, _), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_DUPLEX, hs, t2 + 1)
    tx = w // 2 - tw // 2
    hy = int(90 * s)
    cv2.putText(
        combined, header, (tx, hy), cv2.FONT_HERSHEY_DUPLEX, hs, (20, 20, 20), t2 * 3
    )
    cv2.putText(
        combined, header, (tx, hy), cv2.FONT_HERSHEY_DUPLEX, hs, (255, 220, 60), t2
    )

    card_w, card_h = int(300 * s), int(200 * s)
    gap = int(50 * s)
    total = card_w * 3 + gap * 2
    cx_start = w // 2 - total // 2
    cy = h // 2 - card_h // 2 + int(40 * s)

    card_colors = [(50, 40, 160), (30, 110, 50), (160, 70, 20)]
    keys = ["1", "2", "3"]

    for i in range(min(3, len(top3_labels))):
        x = cx_start + i * (card_w + gap)

        cv2.rectangle(combined, (x, cy), (x + card_w, cy + card_h), (25, 25, 35), -1)
        cv2.rectangle(
            combined, (x, cy), (x + card_w, cy + max(2, int(5 * s))), card_colors[i], -1
        )
        cv2.rectangle(combined, (x, cy), (x + card_w, cy + card_h), (100, 100, 120), 1)

        r = int(22 * s)
        ccx, ccy = x + int(30 * s), cy + int(35 * s)
        cv2.circle(combined, (ccx, ccy), r, card_colors[i], -1)
        cv2.circle(combined, (ccx, ccy), r, (200, 200, 200), 1)
        (kw, kh), _ = cv2.getTextSize(keys[i], cv2.FONT_HERSHEY_DUPLEX, 1.0 * s, t2)
        cv2.putText(
            combined,
            keys[i],
            (ccx - kw // 2, ccy + kh // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * s,
            (255, 255, 255),
            t2,
        )

        label = top3_labels[i]
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.05 * s, t2)
        cv2.putText(
            combined,
            label,
            (x + card_w // 2 - lw // 2, cy + int(115 * s)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.05 * s,
            (240, 240, 240),
            t2,
        )

        prob = top3_probs[i]
        bar_x, bar_y = x + int(20 * s), cy + int(145 * s)
        bar_max = card_w - int(40 * s)
        bar_h = max(5, int(14 * s))
        cv2.rectangle(
            combined, (bar_x, bar_y), (bar_x + bar_max, bar_y + bar_h), (50, 50, 60), -1
        )
        cv2.rectangle(
            combined,
            (bar_x, bar_y),
            (bar_x + int(bar_max * prob), bar_y + bar_h),
            card_colors[i],
            -1,
        )

        pct = f"{prob * 100:.0f}%"
        (pw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, t1)
        cv2.putText(
            combined,
            pct,
            (x + card_w // 2 - pw // 2, cy + int(178 * s)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * s,
            (180, 180, 180),
            t1,
        )

    if typing_mode:
        box_w, box_h = int(500 * s), int(60 * s)
        bx = w // 2 - box_w // 2
        by = cy + card_h + int(30 * s)
        cv2.rectangle(combined, (bx, by), (bx + box_w, by + box_h), (25, 25, 35), -1)
        cv2.rectangle(combined, (bx, by), (bx + box_w, by + box_h), (255, 220, 60), t1)
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        cv2.putText(
            combined,
            typed_text + cursor,
            (bx + int(15 * s), by + int(42 * s)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9 * s,
            (255, 255, 255),
            t2,
        )
        inst = "Type the category   |   ENTER = save   |   ESC = cancel"
    else:
        inst = (
            "1 / 2 / 3 to pick   |   4 = type it   |   ESC = top pick   |   C = redraw"
        )

    (iw, _), _ = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.65 * s, t1)
    cv2.putText(
        combined,
        inst,
        (w // 2 - iw // 2, h - int(28 * s)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * s,
        (20, 20, 20),
        t1 * 4,
    )
    cv2.putText(
        combined,
        inst,
        (w // 2 - iw // 2, h - int(28 * s)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * s,
        (190, 190, 190),
        t1,
    )


def run_generation(canvas_img, label, result):
    try:
        result["img"] = generate_image(canvas_img, label)
    except Exception as e:
        print(f"Generation failed: {e}")
    result["done"] = True


def draw_generating_screen(combined, w, h, label, start_time, thumb):
    dark = np.zeros_like(combined)
    cv2.addWeighted(dark, 0.65, combined, 0.35, 0, combined)

    s = w / 1280.0
    t2 = max(1, int(round(2 * s)))

    dots = "." * (int((time.time() - start_time) * 2) % 4)
    text = f"Creating your {label}{dots}"
    base = f"Creating your {label}..."
    (tw, _), _ = cv2.getTextSize(base, cv2.FONT_HERSHEY_DUPLEX, 1.3 * s, t2)
    tx = w // 2 - tw // 2
    ty = int(h * 0.3)
    cv2.putText(
        combined, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.3 * s, (20, 20, 20), t2 * 3
    )
    cv2.putText(
        combined, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.3 * s, (255, 220, 60), t2
    )

    if thumb is not None:
        th_h = int(h * 0.4)
        th_w = int(th_h * w / h)
        small = cv2.resize(thumb, (th_w, th_h))
        x0 = w // 2 - th_w // 2
        y0 = int(h * 0.38)
        cv2.rectangle(
            combined,
            (x0 - 3, y0 - 3),
            (x0 + th_w + 3, y0 + th_h + 3),
            (255, 220, 60),
            2,
        )
        combined[y0 : y0 + th_h, x0 : x0 + th_w] = small

    angle = int((time.time() - start_time) * 360) % 360
    cx, cy = w // 2, int(h * 0.88)
    r = int(18 * s)
    cv2.ellipse(
        combined, (cx, cy), (r, r), 0, angle, angle + 270, (255, 220, 60), t2 + 1
    )


def draw_result_screen(combined, w, h, result_img, label):
    dark = np.zeros_like(combined)
    cv2.addWeighted(dark, 0.65, combined, 0.35, 0, combined)

    s = w / 1280.0
    t2 = max(1, int(round(2 * s)))
    t1 = max(1, int(round(1 * s)))

    header = "SHAKALAKA!"
    (tw, _), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_DUPLEX, 1.6 * s, t2)
    tx = w // 2 - tw // 2
    hy = int(70 * s)
    cv2.putText(
        combined,
        header,
        (tx, hy),
        cv2.FONT_HERSHEY_DUPLEX,
        1.6 * s,
        (20, 20, 20),
        t2 * 3,
    )
    cv2.putText(
        combined, header, (tx, hy), cv2.FONT_HERSHEY_DUPLEX, 1.6 * s, (60, 220, 255), t2
    )

    if result_img is not None:
        img_h = int(h * 0.6)
        img_w = int(img_h * result_img.shape[1] / result_img.shape[0])
        if img_w > int(w * 0.8):
            img_w = int(w * 0.8)
            img_h = int(img_w * result_img.shape[0] / result_img.shape[1])
        resized = cv2.resize(result_img, (img_w, img_h))
        x0 = w // 2 - img_w // 2
        y0 = h // 2 - img_h // 2 + int(20 * s)
        cv2.rectangle(
            combined,
            (x0 - 4, y0 - 4),
            (x0 + img_w + 4, y0 + img_h + 4),
            (255, 220, 60),
            max(2, int(4 * s)),
        )
        combined[y0 : y0 + img_h, x0 : x0 + img_w] = resized

    inst = f"Your {label}!   |   C = draw again"
    (iw, _), _ = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, t1)
    cv2.putText(
        combined,
        inst,
        (w // 2 - iw // 2, h - int(25 * s)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * s,
        (20, 20, 20),
        t1 * 4,
    )
    cv2.putText(
        combined,
        inst,
        (w // 2 - iw // 2, h - int(25 * s)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * s,
        (230, 230, 230),
        t1,
    )


def draw_hud(combined, w, h, eraser_mode, categories):
    color = (0, 0, 220) if eraser_mode else (0, 215, 255)
    cv2.rectangle(combined, (0, 0), (w - 1, h - 1), color, 6)

    mode_text = "ERASE" if eraser_mode else "DRAW"
    badge_color = (0, 0, 180) if eraser_mode else (0, 150, 200)
    cv2.rectangle(combined, (10, 10), (130, 50), badge_color, -1)
    cv2.putText(
        combined, mode_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2
    )

    panel_x = w - 155
    cv2.rectangle(
        combined, (panel_x - 5, 5), (w - 5, len(categories) * 22 + 15), (20, 20, 40), -1
    )
    for idx, cat in enumerate(categories):
        cv2.putText(
            combined,
            cat,
            (panel_x, 22 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="assets/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

t0 = time.time()

with open("models/categories_15.txt") as f:
    categories = f.read().splitlines()
print(f"[LOAD] categories: {time.time() - t0:.2f}s")

pretrained_model = build_model(num_classes=len(categories))
pretrained_model.load_state_dict(
    torch.load("models/quickDraw_efficientnet_15cat_aug.pth", map_location="cpu")
)
pretrained_model.eval()
print(f"[LOAD] pretrained_model: {time.time() - t0:.2f}s")

scratch_model = QuickDNN(num_classes=len(categories))
scratch_model.load_state_dict(
    torch.load("models/quickDraw_model_15cat.pth", map_location="cpu")
)
scratch_model.eval()
print(f"[LOAD] scratch_model: {time.time() - t0:.2f}s")

pencil_img = cv2.imread("assets/shakalaka_pencil.png", cv2.IMREAD_UNCHANGED)
eraser_img = cv2.imread("assets/eraser.png", cv2.IMREAD_UNCHANGED)
bg_img = cv2.imread("assets/background.jpg")
logo_img = cv2.imread("assets/logo_shakalaka.png", cv2.IMREAD_UNCHANGED)
left_hand_img = cv2.imread("assets/left_hand.png", cv2.IMREAD_UNCHANGED)
right_hand_img = cv2.imread("assets/right_hand.png", cv2.IMREAD_UNCHANGED)
print(f"[LOAD] assets: {time.time() - t0:.2f}s")

cap = cv2.VideoCapture(0)
print(f"[LOAD] camera: {time.time() - t0:.2f}s")
cv2.namedWindow("Shakalaka", cv2.WINDOW_NORMAL)

app_state = "TITLE"
target_hand = None
thumbs_up_hand = None
ret, frame = cap.read()
h, w, d = frame.shape
cv2.resizeWindow("Shakalaka", 1600, int(1600 * h / w))
print(f"[LOAD] first frame: {time.time() - t0:.2f}s")

top3_labels = []
top3_probs_list = []
flash_start_time = 0.0
selected_label = ""
typing_mode = False
typed_text = ""
gen_state = {"img": None, "done": False}
gen_start_time = 0.0
gen_thumb = None

pygame.mixer.init()
print(f"[LOAD] pygame: {time.time() - t0:.2f}s")
predicted_label = ""
canvas = np.zeros((h, w, 3), dtype=np.uint8)
prev_point = None
smoothing_buffer = deque(maxlen=8)
eraser_mode = False
locked_angle = None
prediction_done = False
drawing_sound_playing = False

print(f"[LOAD] HandLandmarker init: {time.time() - t0:.2f}s")
print("[LOAD] entering main loop")
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if app_state == "TITLE":
            draw_title_screen(frame, w, h)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                target_hand = "Left"
                thumbs_up_hand = "Right"
                app_state = "DRAWING"
            elif key == ord("l"):
                target_hand = "Right"
                thumbs_up_hand = "Left"
                app_state = "DRAWING"
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                handedness_label = result.handedness[i][0].category_name

                for start, end in HAND_CONNECTIONS:
                    x1, y1 = (
                        int(hand_landmarks[start].x * w),
                        int(hand_landmarks[start].y * h),
                    )
                    x2, y2 = (
                        int(hand_landmarks[end].x * w),
                        int(hand_landmarks[end].y * h),
                    )
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                if handedness_label == thumbs_up_hand:
                    thumbs_up_curr = is_thumbs_up(hand_landmarks)
                    print(f"thumbs_up: {thumbs_up_curr}")
                    if (
                        thumbs_up_curr
                        and not prediction_done
                        and app_state == "DRAWING"
                    ):
                        preprocessed = preprocess_canvas(canvas.copy())
                        if preprocessed is not None:
                            cv2.imwrite(
                                "debug_preprocessed.png",
                                (preprocessed[0, 0] * 255)
                                .clip(0, 255)
                                .astype(np.uint8),
                            )
                            tensor = torch.tensor(preprocessed, dtype=torch.float32)
                            with torch.no_grad():
                                output = pretrained_model(tensor)
                                probs = torch.softmax(output, dim=1)[0]
                                top3 = torch.topk(probs, 3)
                                top3_labels = [categories[idx] for idx in top3.indices]
                                top3_probs_list = [
                                    probs[i].item() for i in top3.indices
                                ]
                                predicted_label = " | ".join(
                                    f"{categories[i]} ({probs[i] * 100:.0f}%)"
                                    for i in top3.indices
                                )
                            print(f"Predicted: {predicted_label}")
                            prediction_done = True
                            app_state = "PREDICTING"
                            flash_start_time = time.time()
                            cv2.imwrite("debug_canvas.png", canvas)

                if handedness_label != target_hand:
                    continue

                if (
                    app_state == "DRAWING"
                    and not prediction_done
                    and is_pencil_grip(hand_landmarks)
                ):
                    px = int((hand_landmarks[8].x + hand_landmarks[12].x) / 2 * w)
                    py = int(
                        (hand_landmarks[8].y + hand_landmarks[12].y + 0.003) / 2 * h
                    )
                    if locked_angle is None:
                        dx = hand_landmarks[5].x - hand_landmarks[0].x
                        dy = hand_landmarks[5].y - hand_landmarks[0].y
                        locked_angle = math.degrees(math.atan2(dy, dx)) + 135

                    original_frame = frame.copy()
                    if eraser_mode:
                        ex = int((hand_landmarks[8].x + 0.01) * w)
                        ey = int((hand_landmarks[8].y + 0.04) * h)
                        draw_pencil(
                            frame,
                            (ex, ey),
                            eraser_img,
                            locked_angle,
                            size=int(w * 0.09),
                        )
                    else:
                        draw_pencil(
                            frame,
                            (px, py),
                            pencil_img,
                            locked_angle,
                            size=int(w * 0.18),
                        )
                    draw_thumb(frame, original_frame, hand_landmarks, w, h)
                    raw_point = (
                        int(hand_landmarks[8].x * w),
                        int(hand_landmarks[8].y * h),
                    )
                    smoothing_buffer.append(raw_point)
                    curr_point = (
                        int(
                            sum(p[0] for p in smoothing_buffer) / len(smoothing_buffer)
                        ),
                        int(
                            sum(p[1] for p in smoothing_buffer) / len(smoothing_buffer)
                        ),
                    )

                    if prev_point is None:
                        prev_point = curr_point
                    else:
                        dist = np.linalg.norm(
                            np.array(curr_point) - np.array(prev_point)
                        )
                        if dist > 12:
                            if not drawing_sound_playing:
                                sound_file = (
                                    "assets/pencil.mp3"
                                    if eraser_mode is False
                                    else "assets/eraser.mp3"
                                )
                                pygame.mixer.music.load(sound_file)
                                pygame.mixer.music.play(-1)
                                drawing_sound_playing = True
                            drawline(canvas, prev_point, curr_point, w, h, eraser_mode)
                            prev_point = curr_point
                else:
                    prev_point = None
                    smoothing_buffer.clear()
                    locked_angle = None
                    if drawing_sound_playing:
                        pygame.mixer.music.stop()
                        drawing_sound_playing = False
        else:
            prev_point = None
            smoothing_buffer.clear()
            if drawing_sound_playing:
                pygame.mixer.music.stop()
                drawing_sound_playing = False

        combined = cv2.add(frame, canvas)
        if app_state == "DRAWING":
            draw_hud(combined, w, h, eraser_mode, categories)
        elif app_state == "PREDICTING":
            draw_prediction_screen(
                combined,
                w,
                h,
                top3_labels,
                top3_probs_list,
                flash_start_time,
                typing_mode,
                typed_text,
            )
        elif app_state == "GENERATING":
            if gen_state["done"]:
                if gen_state["img"] is not None:
                    app_state = "RESULT"
                else:
                    app_state = "DRAWING"
            draw_generating_screen(
                combined, w, h, selected_label, gen_start_time, gen_thumb
            )
        elif app_state == "RESULT":
            draw_result_screen(combined, w, h, gen_state["img"], selected_label)

        cv2.imshow("Shakalaka", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if app_state == "DRAWING":
            if key == ord("c"):
                canvas[:] = 0
                predicted_label = ""
                prediction_done = False
            elif key == ord("e"):
                eraser_mode = not eraser_mode
        elif app_state == "PREDICTING":
            pick = None
            if typing_mode:
                if key == 13:
                    if typed_text.strip():
                        pick = typed_text.strip().lower()
                elif key == 27:
                    typing_mode = False
                    typed_text = ""
                elif key == 8:
                    typed_text = typed_text[:-1]
                elif 32 <= key <= 126:
                    typed_text += chr(key)
            else:
                if key == ord("1") and top3_labels:
                    pick = top3_labels[0]
                elif key == ord("2") and len(top3_labels) > 1:
                    pick = top3_labels[1]
                elif key == ord("3") and len(top3_labels) > 2:
                    pick = top3_labels[2]
                elif key == ord("4"):
                    typing_mode = True
                    typed_text = ""
                elif key == 27 and top3_labels:
                    pick = top3_labels[0]

            if pick is not None:
                selected_label = pick
                save_canvas(canvas, selected_label)
                gen_thumb = canvas.copy()
                gen_state = {"img": None, "done": False}
                threading.Thread(
                    target=run_generation,
                    args=(canvas.copy(), selected_label, gen_state),
                    daemon=True,
                ).start()
                gen_start_time = time.time()
                canvas[:] = 0
                predicted_label = ""
                prediction_done = False
                top3_labels = []
                top3_probs_list = []
                typing_mode = False
                typed_text = ""
                app_state = "GENERATING"
            elif not typing_mode and key == ord("c"):
                canvas[:] = 0
                predicted_label = ""
                prediction_done = False
                top3_labels = []
                top3_probs_list = []
                app_state = "DRAWING"
        elif app_state == "RESULT":
            if key == ord("c"):
                gen_state = {"img": None, "done": False}
                gen_thumb = None
                app_state = "DRAWING"

cap.release()
cv2.destroyAllWindows()
