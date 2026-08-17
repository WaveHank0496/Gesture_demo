import cv2
import csv
import os
from src.gesture_demo.camera import Camera
from src.gesture_demo.detector import HandDetector

LABEL_MAP = {
    '1': "fist",
    '2': "open",
    '3': "point",
    '4': "yeah",
    '5': "thumb_up",
    '6': "three",
    '7': "phone",
    '8': "ok",
}

OUTPUT_PATH = "data/raw/gestures.csv"


def main():
    camera = Camera(0)
    detector = HandDetector()

    current_label = None
    recording = False
    buffer = []
    session_id = 0          # 新增:遞增計數器,每次存檔前 +1

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=== 資料採集 ===")
    print("數字鍵 1~8: 選手勢 | 空白鍵: 開始/暫停錄 | s: 存檔 | q: 離開")
    print("位置 距離 角度 綜合")

    while True:
        frame = camera.read()
        hands = detector.detect(frame)

        if recording and hands:
            landmarks = hands[0].landmarks
            row = [current_label, session_id]      # label, session_id 放最前面
            for (x, y, z) in landmarks:
                row.extend([x, y, z])
            buffer.append(row)

        status = "REC" if recording else "PAUSE"
        color = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(frame, f"[{status}] label:{current_label} session:{session_id}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"buffer: {len(buffer)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        hand_txt = "hand: YES" if hands else "hand: NO"
        cv2.putText(frame, hand_txt, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            recording = not recording
        elif key == ord('s'):
            _save(buffer)
            buffer = []
            session_id += 1          # 存一次檔 = 結束一個 session,下次遞增
        elif chr(key) in LABEL_MAP if key != 255 else False:
            current_label = LABEL_MAP[chr(key)]
            recording = False
            print(f"切換到: {current_label}")

    camera.release()
    cv2.destroyAllWindows()


def _save(buffer):
    if not buffer:
        print("buffer 是空的,沒東西可存")
        return
    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(buffer)
    print(f"已存 {len(buffer)} 筆到 {OUTPUT_PATH} (session {buffer[0][1]})")


if __name__ == "__main__":
    main()