import math
import os
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

with open("models/categories_15.txt") as f:
    categories = f.read().splitlines()

pretrained_model = build_model(num_classes=len(categories))
pretrained_model.load_state_dict(
    torch.load("models/quickDraw_efficientnet_15cat_aug.pth", map_location="cpu")
)
pretrained_model.eval()

scratch_model = QuickDNN(num_classes=len(categories))
scratch_model.load_state_dict(
    torch.load("models/quickDraw_model_15cat.pth", map_location="cpu")
)
scratch_model.eval()

pencil_img = cv2.imread("assets/shakalaka_pencil.png", cv2.IMREAD_UNCHANGED)
eraser_img = cv2.imread("assets/eraser.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

# Hand selection screen
target_hand = None
thumbs_up_hand = None
ret, frame = cap.read()
h, w, d = frame.shape

top3_labels = []
waiting_for_confirm = False

while target_hand is None:
    ret, frame = cap.read()
    cv2.putText(
        frame,
        "Which hand will you draw with?",
        (30, h // 2 - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Press R for Right hand  |  L for Left hand",
        (30, h // 2 + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.imshow("Shakalaka", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        target_hand = "Left"  # MediaPipe mirrors: user's Right = "Left"
        thumbs_up_hand = "Right"  # non-drawing hand
    elif key == ord("l"):
        target_hand = "Right"  # MediaPipe mirrors: user's Left = "Right"
        thumbs_up_hand = "Left"  # non-drawing hand

pygame.mixer.init()
predicted_label = ""
canvas = np.zeros((h, w, 3), dtype=np.uint8)
prev_point = None
smoothing_buffer = deque(maxlen=8)
eraser_mode = False
locked_angle = None
prediction_done = False
drawing_sound_playing = False

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
                    if thumbs_up_curr and not prediction_done:
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
                                predicted_label = " | ".join(
                                    f"{categories[i]} ({probs[i] * 100:.0f}%)"
                                    for i in top3.indices
                                )
                            print(f"Predicted: {predicted_label}")
                            prediction_done = True
                            waiting_for_confirm = True
                            cv2.imwrite("debug_canvas.png", canvas)

                if handedness_label != target_hand:
                    continue

                if not prediction_done and is_pencil_grip(hand_landmarks):
                    px = int(
                        (hand_landmarks[8].x + hand_landmarks[12].x - 0.02) / 2 * w
                    )
                    py = int(
                        (hand_landmarks[8].y + hand_landmarks[12].y + 0.003) / 2 * h
                    )
                    if locked_angle is None:
                        dx = hand_landmarks[5].x - hand_landmarks[0].x
                        dy = hand_landmarks[5].y - hand_landmarks[0].y
                        locked_angle = math.degrees(math.atan2(dy, dx)) + 135

                    original_frame = frame.copy()
                    if eraser_mode:
                        ex = int((hand_landmarks[8].x - 0.01) * w)
                        ey = int((hand_landmarks[8].y + 0.04) * h)
                        draw_pencil(frame, (ex, ey), eraser_img, locked_angle, size=50)
                    else:
                        draw_pencil(frame, (px, py), pencil_img, locked_angle, size=150)
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

        if predicted_label:
            for j, part in enumerate(predicted_label.split(" | ")):
                cv2.putText(
                    combined,
                    part,
                    (30, 50 + j * 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )
            if waiting_for_confirm:
                cv2.putText(
                    combined,
                    "Press 1/2/3 to confirm | 4 to type | ESC = top pick",
                    (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Shakalaka", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
            predicted_label = ""
            prediction_done = False
        elif key == ord("e"):
            eraser_mode = not eraser_mode

        if waiting_for_confirm:
            if key == ord("1"):
                save_canvas(canvas, top3_labels[0])
                # generated = generate_image(canvas, top3_labels[0])
                # if generated is not None:
                #     cv2.imshow("Generated", generated)
                #     cv2.waitKey(0)
            elif key == ord("2"):
                save_canvas(canvas, top3_labels[1])
                # generated = generate_image(canvas, top3_labels[1])
                # if generated is not None:
                #     cv2.imshow("Generated", generated)
                #     cv2.waitKey(0)
            elif key == ord("3"):
                save_canvas(canvas, top3_labels[2])
                # generated = generate_image(canvas, top3_labels[2])
                # if generated is not None:
                #     cv2.imshow("Generated", generated)
                #     cv2.waitKey(0)
            elif key == ord("4"):
                print("Type category name: ")
                label = input()
                save_canvas(canvas, label)
                # generated = generate_image(canvas, label)
                # if generated is not None:
                #     cv2.imshow("Generated", generated)
                #     cv2.waitKey(0)
            elif key == 27:  # ESC
                save_canvas(canvas, top3_labels[0])
                # generated = generate_image(canvas, top3_labels[0])
                # if generated is not None:
                #     cv2.imshow("Generated", generated)
                #     cv2.waitKey(0)
            else:
                pass

            if key in (ord("1"), ord("2"), ord("3"), ord("4"), 27):
                canvas[:] = 0
                predicted_label = ""
                prediction_done = False
                waiting_for_confirm = False
                top3_labels = []

cap.release()
cv2.destroyAllWindows()
