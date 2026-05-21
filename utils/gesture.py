from math import sqrt

def is_pencil_grip(hand_landmarks, threshold = 0.05):
    ec_distance = sqrt((hand_landmarks.landmark[4].x - hand_landmarks.landmark[8].x) ** 2 + (hand_landmarks.landmark[4].y - hand_landmarks.landmark[8].y) ** 2)
    return ec_distance <= threshold

def is_open_hand(hand_landmarks):
    if (((abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[2].x)) > 0) and ((hand_landmarks.landmark[8].y - hand_landmarks.landmark[5].y) < 0) and ((hand_landmarks.landmark[12].y - hand_landmarks.landmark[9].y) < 0) and ((hand_landmarks.landmark[16].y - hand_landmarks.landmark[13].y) < 0) and ((hand_landmarks.landmark[20].y - hand_landmarks.landmark[17].y) < 0)):
        return True
    else:
        return False
