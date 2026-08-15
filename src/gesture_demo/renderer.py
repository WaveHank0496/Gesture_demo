import cv2
from src.gesture_demo.camera import Camera
from src.gesture_demo.detector import HandDetector
from src.gesture_demo.contracts import HandLandmarks
from src.gesture_demo.smoother import Smoother
from src.gesture_demo.recognizer import recognize
from src.gesture_demo.interaction.trigger import PinchTrigger
from src.gesture_demo.contracts import HandLandmarks, RenderEventType

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (1, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (9, 13), (13, 14), (14, 15), (15, 16), # 無名指
    (13, 17), (17, 18), (18, 19), (19, 20),# 小指
    (0, 17),                                # 手掌根部
]

def draw_landmarks(frame, hands: HandLandmarks):
    # 如果沒有偵測到手 就輸出原始畫面
    if not hands:
        return frame

    # step 1 先把所有的座標點算好存進points
    height, width = frame.shape[:2]
    for hand in hands:
        points = []
        for landmark in hand.landmarks:
            px = int(landmark[0] * width)
            py = int(landmark[1] * height)
            points.append((px, py))

        # step 2 畫線 兩兩端點畫線
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)

        # step 3 畫點
        for point in points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

    return frame

def draw_gesture_text(frame, state):
    if not state.hand_detected:
        return frame

    # 手勢文字(不管哪個手勢,同一行,文字動態取自 enum)
    text = state.gesture.value.upper()
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 捏合值(第二行,y 往下)
    pinch_text = f"pinch: {state.pinch_strength:.2f}"
    cv2.putText(frame, pinch_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return frame

if __name__ == "__main__":
    from src.gesture_demo.camera import Camera

    cam = Camera(0)
    detector = HandDetector()
    smoother = Smoother(alpha=0.5)
    trigger = PinchTrigger()

    while True:
        frame = cam.read()
        hands = detector.detect(frame)
        hands = smoother.smooth(hands)
        state = recognize(hands)
        command = trigger.process(state)

        # 驗證:只在真的觸發 CLICK 時印
        if command.event_type == RenderEventType.CLICK:
            print(f"CLICK! 位置:{command.event_position}")

        frame = draw_landmarks(frame, hands)
        frame = draw_gesture_text(frame, state)

        cv2.imshow("Render Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()