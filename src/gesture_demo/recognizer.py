import math
import numpy as np
import torch
from src.gesture_demo.contracts import HandLandmarks, GestureState, Gesture
from src.gesture_demo.features import normalize
from src.gesture_demo.dataset import GESTURE_LABELS
from src.gesture_demo.model import GestureMLP

# 幾何計算
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

def recognize_gesture_by_rules(landmarks) -> Gesture:
    fingers = {name: is_finger_extended(landmarks, tid) for name, tid in FINGER_TIPS.items()}
    if not any(fingers.values()):
        return Gesture.FIST
    if fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
        return Gesture.OPEN
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
    return Gesture.NONE

# ── 主辨識器:雙模式 ──────────────────────────
class GestureRecognizer:
    def __init__(self, model_path="models/gesture_mlp.pth"):
        self.mode = "ml"                         # "ml" 或 "rules"
        self.model = GestureMLP()
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except (FileNotFoundError, RuntimeError) as e:
            # 最常見的原因:新增手勢後類別數變了,舊的 .pth 對不上 → 要重新訓練。
            # 這裡不讓整個 app 掛掉,先退回規則模式(但規則模式只認得舊的幾個手勢)。
            print(f"[Recognizer] 載入 {model_path} 失敗:{e}")
            print(f"[Recognizer] 目前類別數 = {len(GESTURE_LABELS)}。"
                  "若是剛新增手勢,請先錄資料再跑 train 重新產生 .pth。")
            print("[Recognizer] 暫時切換到規則模式(rules)。")
            self.model = None
            self.mode = "rules"

    def toggle(self):
        if self.model is None and self.mode == "rules":
            print("[Recognizer] 模型沒載入成功,無法切到 ml 模式(請先重新訓練)")
            return
        self.mode = "rules" if self.mode == "ml" else "ml"
        print(f"[Recognizer] 切換到: {self.mode}")

    def _predict_by_model(self, landmarks) -> Gesture:
        landmarks_2d = [(p[0], p[1]) for p in landmarks]   # 先砍成 2D
        normalized = normalize(landmarks_2d)
        flat = []
        for (x, y) in normalized:
            flat.extend([x, y])
        x = torch.tensor(np.array([flat]), dtype=torch.float32)
        with torch.no_grad():
            idx = self.model(x).argmax(dim=1).item()
        return Gesture(GESTURE_LABELS[idx])

    def predict_gesture(self, landmarks) -> Gesture:
        if self.mode == "ml":
            return self._predict_by_model(landmarks)
        else:
            return recognize_gesture_by_rules(landmarks)

    def recognize(self, hands: list[HandLandmarks]) -> GestureState:
        if not hands:
            return GestureState(
                hand_detected=False, gesture=Gesture.NONE,
                pinch_strength=0.0, hand_position=(0.0, 0.0), index_tip=(0.0, 0.0),
            )
        landmarks = hands[0].landmarks
        return GestureState(
            hand_detected=True,
            gesture=self.predict_gesture(landmarks),
            pinch_strength=compute_pinch(landmarks),
            hand_position=(landmarks[0][0], landmarks[0][1]),
            index_tip=(landmarks[8][0], landmarks[8][1]),
        )


# ── 模組層級單例 + 轉發函式 ──────────────────
_recognizer = GestureRecognizer()


def recognize(hands: list[HandLandmarks]) -> GestureState:
    return _recognizer.recognize(hands)


def toggle_mode():                # 對外暴露切換,給 app.py 呼叫
    _recognizer.toggle()


def get_mode() -> str:            # 讓 app 能顯示現在是哪個模式
    return _recognizer.mode


"""
# 原邏輯
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
"""
