from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType, Gesture
from src.gesture_demo.sound import SoundPlayer

GESTURE_IMAGES = {
    Gesture.FIST: "fist.jpg",
    Gesture.OPEN: "open.jpg",
    Gesture.POINT: "point.jpg",
    Gesture.YEAH: "yeah.jpg",
    Gesture.THUMB_UP: "thumb_up.jpg",
    Gesture.THREE: "three.jpg",
    Gesture.FOUR: "four.jpg",
}

GESTURE_SOUNDS = {
    Gesture.FIST: "fist.wav",
    Gesture.OPEN: "open.wav",
    Gesture.POINT: "point.wav",
    Gesture.YEAH: "yeah.wav",
    Gesture.THUMB_UP: "thumb_up.wav",
    Gesture.THREE: "three.wav",
    Gesture.FOUR: "four.wav",
}


class GestureImage(Interaction):
    def __init__(self):
        self.last_gesture = Gesture.NONE      # 上一幀手勢(邊緣偵測用)
        self.sound_player = SoundPlayer()

    def process(self, state: GestureState) -> RenderCommand:
        image_name = None
        current = Gesture.NONE

        if state.hand_detected:
            current = state.gesture
            image_name = GESTURE_IMAGES.get(state.gesture, None)

            # 邊緣偵測:手勢「剛變成」新手勢那一瞬間,播一次音效
            if current != self.last_gesture:
                sound = GESTURE_SOUNDS.get(current, None)
                if sound:
                    self.sound_player.play(sound)

        self.last_gesture = current            # 更新(沒手時是 NONE)

        return RenderCommand(
            event_type=RenderEventType.NONE,
            event_position=(0.0, 0.0),
            image_name=image_name,
        )