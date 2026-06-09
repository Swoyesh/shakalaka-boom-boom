import math

import cv2
import numpy as np


def draw_pencil(frame, position, pencil_img, angle: float = 0.0, size=100):
    h, w = pencil_img.shape[:2]
    scale = size / h
    new_w = int(w * scale)
    pencil_resized = cv2.resize(pencil_img, (new_w, size))

    # Grip point within the pencil image — ~60% down (below the face, mid-body)
    grip_x = new_w // 2
    grip_y = int(size * 0.85)

    # Large canvas so rotation never clips the pencil
    diag = int(math.sqrt(new_w**2 + size**2)) + 20
    canvas_size = diag * 2
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    # Place pencil so grip point sits at canvas center
    x_off = canvas_size // 2 - grip_x
    y_off = canvas_size // 2 - grip_y

    src_x1 = max(0, -x_off)
    src_y1 = max(0, -y_off)
    src_x2 = min(new_w, canvas_size - x_off)
    src_y2 = min(size, canvas_size - y_off)
    dst_x1 = max(0, x_off)
    dst_y1 = max(0, y_off)

    canvas[dst_y1 : dst_y1 + (src_y2 - src_y1), dst_x1 : dst_x1 + (src_x2 - src_x1)] = (
        pencil_resized[src_y1:src_y2, src_x1:src_x2]
    )

    # Rotate canvas around its center, which is the grip point
    center = (canvas_size // 2, canvas_size // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        M,
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Place canvas so its center (grip point) lands exactly at (px, py)
    px, py = position
    x1 = px - canvas_size // 2
    y1 = py - canvas_size // 2
    x2 = x1 + canvas_size
    y2 = y1 + canvas_size

    fh, fw = frame.shape[:2]

    src_cx1 = max(0, -x1)
    src_cy1 = max(0, -y1)
    src_cx2 = canvas_size - max(0, x2 - fw)
    src_cy2 = canvas_size - max(0, y2 - fh)

    if src_cx2 <= src_cx1 or src_cy2 <= src_cy1:
        return frame

    dst_cx1 = max(0, x1)
    dst_cy1 = max(0, y1)
    dst_cx2 = dst_cx1 + (src_cx2 - src_cx1)
    dst_cy2 = dst_cy1 + (src_cy2 - src_cy1)

    bgr = rotated[src_cy1:src_cy2, src_cx1:src_cx2, :3]
    alpha = rotated[src_cy1:src_cy2, src_cx1:src_cx2, 3].astype(np.uint8)
    alpha_inv = cv2.bitwise_not(alpha)

    roi = frame[dst_cy1:dst_cy2, dst_cx1:dst_cx2]
    pencil_fg = cv2.bitwise_and(bgr, bgr, mask=alpha)
    bg = cv2.bitwise_and(roi, roi, mask=alpha_inv)
    frame[dst_cy1:dst_cy2, dst_cx1:dst_cx2] = cv2.add(bg, pencil_fg)

    return frame
