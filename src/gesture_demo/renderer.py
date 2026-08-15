import cv2
from src.gesture_demo.contracts import HandLandmarks, GestureState, RenderCommand, RenderEventType

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

CLICK_DURATION = 15      # CLICK 圈停留幾幀


class Renderer:
    def __init__(self):
        self.click_pos = None      # 最近 CLICK 的位置(像素)
        self.click_timer = 0       # 還要畫幾幀

    def render(self, frame, hands, state, command):
        # 1. 畫手骨架
        self._draw_landmarks(frame, hands)
        # 2. 畫手勢文字
        self._draw_gesture_text(frame, state)
        # 3. 處理並畫 CLICK 效果
        self._draw_click(frame, command)
        return frame

    def _draw_landmarks(self, frame, hands):
        if not hands:
            return
        height, width = frame.shape[:2]
        for hand in hands:
            points = []
            for landmark in hand.landmarks:
                px = int(landmark[0] * width)
                py = int(landmark[1] * height)
                points.append((px, py))
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
            for point in points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

    def _draw_gesture_text(self, frame, state):
        if not state.hand_detected:
            return
        cv2.putText(frame, state.gesture.value.upper(), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"pinch: {state.pinch_strength:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    def _draw_click(self, frame, command):
        height, width = frame.shape[:2]
        # 收到新的 CLICK:記錄位置、重設計時器
        if command.event_type == RenderEventType.CLICK:
            x = int(command.event_position[0] * width)
            y = int(command.event_position[1] * height)
            self.click_pos = (x, y)
            self.click_timer = CLICK_DURATION
        # 還在停留期間:畫圈,計時器遞減
        if self.click_timer > 0 and self.click_pos is not None:
            cv2.circle(frame, self.click_pos, 40, (0, 0, 255), 3)   # 紅色空心圈
            self.click_timer -= 1