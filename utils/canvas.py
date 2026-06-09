import cv2


def drawline(canvas, prev_point, curr_point, w, h, eraser=False):
    color = (0, 0, 0) if eraser else (255, 255, 255)
    thickness = 30 if eraser else 2
    offset_x = int(-0.01 * w)
    offset_y = int(0.01 * h)
    final_pre_point = (prev_point[0] + offset_x, prev_point[1] + offset_y)
    final_curr_point = (curr_point[0] + offset_x, curr_point[1] + offset_y)
    if prev_point is not None and not eraser:
        cv2.line(
            canvas,
            final_pre_point,
            final_curr_point,
            color,
            thickness,
        )
    elif prev_point is not None and eraser:
        cv2.line(
            canvas,
            prev_point,
            curr_point,
            color,
            thickness,
        )
