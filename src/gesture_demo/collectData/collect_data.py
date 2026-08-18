import cv2
import csv
import os
from src.gesture_demo.camera import Camera
from src.gesture_demo.detector import HandDetector

# 已經錄過、模型也訓練過的手勢 → 沿用原本的數字鍵。
# 留著不刪:之後想針對某個舊手勢「補錄」更多角度的資料時,直接按數字就能錄,不用改 code。
TRAINED_LABEL_MAP = {
    '1': "fist",
    '2': "open",
    '3': "point",
    '4': "yeah",
    '5': "thumb_up",
    '6': "three",
    '7': "phone",
    '8': "ok",
}

# 新增、還沒有資料的手勢 → 數字鍵不夠用了,改用鍵盤上排字母(照 qwerty 順序)。
# 已被佔用的鍵不能用:q = 離開、s = 存檔、空白鍵 = 開始/暫停錄影,所以從 w 開始排。
NEW_LABEL_MAP = {
    'w': "four",
    'e': "seven",
    'r': "eight",
    't': "gun",
    'y': "split",
    'u': "rock",
    'i': "middle",
}

# 實際生效的對應表(兩張表合起來,15 個手勢隨時都能錄)
LABEL_MAP = {**TRAINED_LABEL_MAP, **NEW_LABEL_MAP}

OUTPUT_PATH = "data/raw/gestures.csv"


def _next_session_id() -> int:
    """接著 csv 裡最大的 session_id 往下編。

    session_id 是切分 train/test 的單位。如果每次重跑都從 0 開始,
    第二次錄的 session 0 會跟第一次的 session 0 被當成同一段錄製,
    分層切分就會變粗。接著編號可以保證每段錄製都是獨立的一段。
    """
    if not os.path.exists(OUTPUT_PATH):
        return 0
    max_id = -1
    with open(OUTPUT_PATH, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                max_id = max(max_id, int(row[1]))
            except ValueError:
                continue
    return max_id + 1


def main():
    camera = Camera(0)
    detector = HandDetector()

    current_label = None
    recording = False
    buffer = []
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    session_id = _next_session_id()   # 遞增計數器,接著既有資料往下編,每次存檔後 +1

    print("=== 資料採集 ===")
    print("空白鍵: 開始/暫停錄 | s: 存檔(結束一個 session) | q: 離開")
    print("--- 選手勢 ---")
    print("舊手勢(已有資料,要補錄才按):"
          + "  ".join(f"{k}={v}" for k, v in TRAINED_LABEL_MAP.items()))
    print("新手勢(這次要錄的):"
          + "  ".join(f"{k}={v}" for k, v in NEW_LABEL_MAP.items()))
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
        cv2.putText(frame, "new: " + " ".join(f"{k}={v}" for k, v in NEW_LABEL_MAP.items()),
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
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