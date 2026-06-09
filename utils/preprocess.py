import cv2
import numpy as np

def preprocess_canvas(snapshot):
    grayscaled_snapshot = cv2.cvtColor(snapshot, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero(grayscaled_snapshot)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    cropped = grayscaled_snapshot[y:y+h, x:x+w]
    binary = (cropped > 0).astype(np.uint8) * 255
    thickened = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations = 2)
    scale = 224 / max(w, h)
    new_h = int(scale * h)
    new_w = int(scale * w)
    resized = cv2.resize(thickened, (new_w, new_h))
    padded = np.zeros((224, 224), dtype=np.uint8)
    pad_top = (224 - new_h) // 2
    pad_left = (224 - new_w) // 2
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    three_padded = np.stack([padded] * 3, axis=0)

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    normalized = (three_padded / 255.0 - mean) / std
    final = normalized.reshape(1, 3, 224, 224)
    return final
