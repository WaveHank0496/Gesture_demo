from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType, Gesture
from src.gesture_demo.sound import SoundPlayer

GESTURE_IMAGES = {
    Gesture.FIST: "oiiiaiii.jpg",  
    Gesture.OPEN: "kenchiana.jpg",
    Gesture.POINT: "weiweimonmon.jpg",
    Gesture.YEAH: "emotional_damage.jpg",
    Gesture.THUMB_UP: "fuhhhh.jpg",
    Gesture.THREE: "coldwater.jpg",
    Gesture.FOUR: "Uncle.jpg",
    Gesture.PHONE: "phone.jpg",
    Gesture.OK: "piyan.jpg",
  
    Gesture.SEVEN: "sixseven.jpg",
    Gesture.EIGHT: "pull_into_trash.jpg",
    Gesture.GUN: "president_goshot.jpg",
    Gesture.SPLIT: "rapgod.jpg",
    Gesture.ROCK: "rickroll.jpg",
    Gesture.MIDDLE: "cowbae.jpg",
}

GESTURE_SOUNDS = {
    Gesture.FIST: "fist.mp3",               # 喔咿咿阿伊
    Gesture.OPEN: "open.mp3",               # 肯洽拿
    Gesture.POINT: "point.mp3",             # 葳葳孟孟
    Gesture.YEAH: "emotional_damage.mp3",   # 尬電
    Gesture.THUMB_UP: "thumb_up.mp3",       # fuhhhhh
    Gesture.THREE: "bao-chi-ding-wang.mp3",         
    Gesture.FOUR: "Uncle_weAreInAccident_uncle.mp3",           
    Gesture.PHONE: "phone.mp3",             # your phone ringing
    Gesture.OK: "piyan.mp3",

    Gesture.SEVEN: "six-seven.mp3",
    Gesture.EIGHT: "pull_into_trash_truck.mp3",
    Gesture.GUN: "president_you_inherent_rrrrr.mp3",
    Gesture.SPLIT: "rapgod.mp3",
    Gesture.ROCK: "rickroll.mp3",
    Gesture.MIDDLE: "cowbae.mp3",
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
        else:
            self.sound_player.stop()      # ← 沒手:停掉音效

        self.last_gesture = current            # 更新(沒手時是 NONE)

        return RenderCommand(
            event_type=RenderEventType.NONE,
            event_position=(0.0, 0.0),
            image_name=image_name,
        )