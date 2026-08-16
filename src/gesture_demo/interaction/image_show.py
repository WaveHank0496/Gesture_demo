from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType, Gesture

GESTURE_IMAGES = {
    Gesture.FIST: "fist.jpg",
    Gesture.OPEN: "open.jpg",
    Gesture.POINT: "point.jpg",
    Gesture.YEAH: "yeah.jpg",
    Gesture.THUMB_UP: "thumb_up.jpg",
    Gesture.THREE: "three.jpg",
    Gesture.FOUR: "four.jpg"
}


class GestureImage(Interaction):
    def process(self, state: GestureState) -> RenderCommand:
        image_name = None
        if state.hand_detected:
            image_name = GESTURE_IMAGES.get(state.gesture, None)   # 查不到就 None

        return RenderCommand(
            event_type=RenderEventType.NONE,      # 這個模式不觸發事件
            event_position=(0.0, 0.0),
            image_name=image_name,                # 帶著要顯示的圖片名(或 None)
        )