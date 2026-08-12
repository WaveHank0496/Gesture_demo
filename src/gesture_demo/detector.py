import cv2
import mediapipe as mp
from src.gesture_demo.contracts import HandLandmarks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self):
        # 建立mediapipe的手部偵測器
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')      # 泛型模型檔位置
        options = vision.HandLandmarkerOptions(                                         # 手部模型  
            base_options=base_options, 
            num_hands=1,                            # 先用一隻手
            running_mode=vision.RunningMode.VIDEO   # 用VIDEO mode
        )  
        self.landmarker = vision.HandLandmarker.create_from_options(options)                   # 按照設定把detector造出來
        self.timestamp = 0

    def detect(self, frame) -> HandLandmarks:
        # 收一幀圖像 -> 吐成 handlandmark格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)       # 把numpy資料包成mp.image
        self.timestamp += 1
        result = self.landmarker.detect_for_video(mp_image, self.timestamp)

        if not result.hand_landmarks:       # 就是判斷他是否empty 因為empty 那個landmark的list為空=False
            return HandLandmarks(hand_detected=False, landmarks=[])

        hand = result.hand_landmarks[0]     # hand_landmarks抓的是很多隻手 目前只有追蹤一隻
        points = []
        for point in hand:
            points.append((point.x, point.y, point.z))

        return HandLandmarks(hand_detected=True, landmarks=points)


if __name__ == "__main__":
    from src.gesture_demo.camera import Camera

    cam = Camera(0)
    detector = HandDetector()

    while True:
        frame = cam.read()
        result = detector.detect(frame)

        if result.hand_detected:
            print(f"偵測到手，共{len(result.landmarks)}個點，第一個點:{result.landmarks[0]}")
        else:
            print("未偵測到手")

        cv2.imshow("Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
