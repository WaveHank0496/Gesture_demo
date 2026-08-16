import math
from src.gesture_demo.contracts import HandLandmarks, GestureState, Gesture

def distance(p1, p2):
    # p1, p2 是 (x, y, z) tuple,算它們的 2D 距離(只用 x, y) 直接算歐氏距離
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def is_finger_extended(landmarks, tip_id, threshold=1.5):
    # landmarks: 一隻手的 21 個點 (list of tuple)
    # tip_id: 指尖的編號(食指=8, 中指=12...)
    # 回傳 True(伸直) / False(彎曲)
    standard = distance(landmarks[0], landmarks[9])
    tip_dis = distance(landmarks[tip_id], landmarks[0]) / standard
    if tip_dis > threshold:
        return True
    else:
        return False

def compute_pinch(landmarks):
    palm = distance(landmarks[0], landmarks[9])          # 手掌基準(正規化用)
    pinch_dist = distance(landmarks[4], landmarks[8]) / palm   # 正規化後的捏合距離
    
    # 把距離映射成 0~1 的 strength:距離越小,strength 越大
    max_dist = 1.0          # 超過這個正規化距離就算完全沒捏
    strength = 1.0 - min(pinch_dist / max_dist, 1.0)
    return strength

# 五根指尖編號
FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
def recognize_gesture(landmarks) -> Gesture:
    # 判斷食指到小指四根的伸直狀態
    fingers = {
        name: is_finger_extended(landmarks, tip_id)
        for name, tip_id in FINGER_TIPS.items()
    }

    # 看組合對應哪個手勢
    if not any(fingers.values()):          # 五指全彎
        return Gesture.FIST
    #if all(fingers.values()):              # 五指全伸
    if fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
        return Gesture.OPEN
    #if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
    #    return Gesture.POINT               # 只有食指伸
    # 伸直的手指數量 == 1,且食指是伸直的
    if sum(fingers.values()) == 1 and fingers["index"]:
        return Gesture.POINT
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return Gesture.YEAH
    if sum(fingers.values()) == 1 and fingers["thumb"]:
        return Gesture.THUMB_UP
    if sum(fingers.values()) == 3 and fingers["index"] and fingers["middle"] and fingers["ring"]:
        return Gesture.THREE
    if sum(fingers.values()) == 2 and fingers["thumb"] and fingers["pinky"]:
        return Gesture.PHONE
    return Gesture.NONE                     # 其他情況不歸類

def recognize(hands: list[HandLandmarks]) -> GestureState:
    # 沒偵測到手:回傳一個「空的」GestureState
    if not hands:
        return GestureState(
            hand_detected=False,
            gesture=Gesture.NONE,
            pinch_strength=0.0,
            hand_position=(0.0, 0.0),
            index_tip=(0.0, 0.0),
        )

    # 有手:只處理第一隻手(你的互動用單手)
    landmarks = hands[0].landmarks

    gesture = recognize_gesture(landmarks)
    pinch = compute_pinch(landmarks)

    # 手的位置用手腕(點0),食指尖用點8,只取 x,y
    hand_pos = (landmarks[0][0], landmarks[0][1])
    index_tip = (landmarks[8][0], landmarks[8][1])

    return GestureState(
        hand_detected=True,
        gesture=gesture,
        pinch_strength=pinch,
        hand_position=hand_pos,
        index_tip=index_tip,
    )
