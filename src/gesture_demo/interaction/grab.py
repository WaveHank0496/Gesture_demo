import math
from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType

PINCH_THRESHOLD = 0.7
GRAB_RADIUS = 0.15      # 手離方塊多近算抓得到(正規化距離)


class GrabDrag(Interaction):
    def __init__(self):
        self.box_pos = (0.5, 0.5)    # 方塊位置(畫面中央,正規化座標)
        self.grabbing = False        # 是否正抓著方塊

    def process(self, state: GestureState) -> RenderCommand:
        # 沒手:放開方塊,方塊留在原地
        if not state.hand_detected:
            self.grabbing = False
            return RenderCommand(event_type=RenderEventType.GRAB, event_position=self.box_pos)

        is_pinching = state.pinch_strength > PINCH_THRESHOLD

        if is_pinching:
            if self.grabbing:
                # 已經抓著 → 方塊跟著手移動
                self.box_pos = state.index_tip
            else:
                # 還沒抓著 → 檢查手是否靠近方塊,夠近就抓起
                dist = math.hypot(
                    state.index_tip[0] - self.box_pos[0],
                    state.index_tip[1] - self.box_pos[1],
                )
                if dist < GRAB_RADIUS:
                    self.grabbing = True
        else:
            # 沒捏合 → 放開
            self.grabbing = False

        return RenderCommand(event_type=RenderEventType.GRAB, event_position=self.box_pos)