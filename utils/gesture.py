from math import sqrt

def is_pencil_grip(hand_landmarks, threshold = 0.06):
    ec_distance = sqrt((hand_landmarks[4].x - hand_landmarks[8].x) ** 2 + (hand_landmarks[4].y - hand_landmarks[8].y) ** 2)
    return ec_distance <= threshold

def is_open_hand(hand_landmarks):
    if (((abs(hand_landmarks[4].x - hand_landmarks[2].x)) > 0) and ((hand_landmarks[8].y - hand_landmarks[5].y) < 0) and ((hand_landmarks[12].y - hand_landmarks[9].y) < 0) and ((hand_landmarks[16].y - hand_landmarks[13].y) < 0) and ((hand_landmarks[20].y - hand_landmarks[17].y) < 0)):
        return True
    else:
        return False

def is_thumbs_up(hand_landmarks):
    thumb_highest = (hand_landmarks[4].y < hand_landmarks[8].y and
                     hand_landmarks[4].y < hand_landmarks[12].y and
                     hand_landmarks[4].y < hand_landmarks[16].y and
                     hand_landmarks[4].y < hand_landmarks[20].y)
    thumb_up = hand_landmarks[4].y < hand_landmarks[2].y
    fingers_curled = (hand_landmarks[8].y > hand_landmarks[6].y and
                      hand_landmarks[12].y > hand_landmarks[10].y and
                      hand_landmarks[16].y > hand_landmarks[14].y and
                      hand_landmarks[20].y > hand_landmarks[18].y)
    return thumb_highest and thumb_up and fingers_curled

