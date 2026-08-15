from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType

PINCH_THRESHOLD = 0.7      # 捏合門檻,超過算捏著


class PinchTrigger(Interaction):
    def __init__(self):
        self.was_pinching = False      # 上一幀是否捏著(跨幀狀態)

    def process(self, state: GestureState) -> RenderCommand:
        # 沒手:回傳一個「什麼都不做」的 RenderCommand
        if not state.hand_detected:
            self.was_pinching = False       # 手離開,重置狀態
            return RenderCommand(event_type=RenderEventType.NONE, event_position=(0.0, 0.0))

        # 這一幀是否捏著
        is_pinching = state.pinch_strength > PINCH_THRESHOLD

        command = RenderCommand(event_type=RenderEventType.NONE, event_position=(0.0, 0.0)) # 確保任何情況下都有狀態

        if is_pinching and not self.was_pinching:
            command = RenderCommand(event_type=RenderEventType.CLICK, event_position=state.hand_position)

        self.was_pinching = is_pinching
        return command