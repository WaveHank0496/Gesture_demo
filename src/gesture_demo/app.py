import cv2
from src.gesture_demo.camera import Camera
from src.gesture_demo.detector import HandDetector
from src.gesture_demo.smoother import Smoother
from src.gesture_demo.recognizer import recognize
from src.gesture_demo.renderer import Renderer
from src.gesture_demo.interaction.trigger import PinchTrigger
from src.gesture_demo.interaction.grab import GrabDrag
from src.gesture_demo.interaction.draw import DrawPen
from src.gesture_demo.contracts import RenderEventType

class App:
    def __init__(self):
        # 建立所有模組(組裝線的「零件準備」)
        self.camera = Camera(0)
        self.detector = HandDetector()
        self.smoother = Smoother(alpha=0.5)
        #interactions
        self.interactions = {
            '1' : PinchTrigger(),
            '2' : GrabDrag(),
            '3' : DrawPen(),
        }
        self.interaction = self.interactions['1']
        self.renderer = Renderer()

    def run(self):
        # 主迴圈:把六個模組串起來
        # read → detect → smooth → recognize → process → draw → show
        while True:
            frame = self.camera.read()
            hands = self.detector.detect(frame)
            hands = self.smoother.smooth(hands)
            state = recognize(hands)
            command = self.interaction.process(state)

            # 渲染
            frame = self.renderer.render(frame, hands, state, command)
            cv2.imshow("Gesture Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255 and chr(key) in self.interactions:      # 先確認有按鍵,才 chr
                self.interaction = self.interactions[chr(key)]

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.run()