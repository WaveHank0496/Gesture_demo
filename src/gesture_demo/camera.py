import cv2

class Camera:
    def __init__(self, camera_id: int = 0):
        # 把攝影機開起來，狀態維持
        self.capture = cv2.VideoCapture(camera_id)

    def read(self):
        # 讀一幀 回傳資料
        success, frame = self.capture.read()
        if not success:
            raise RuntimeError("讀取攝影機失敗")
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        # 關攝影機
        self.capture.release()


# 測試
if __name__ == "__main__":
    cam = Camera(0)
    while True:
        frame = cam.read()
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()