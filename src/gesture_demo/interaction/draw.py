from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType, Gesture


class DrawPen(Interaction):
    def __init__(self):
        self.trail = []      # 累積的軌跡點

    def process(self, state: GestureState) -> RenderCommand:
        # 沒手:不加點,但保留已畫的軌跡
        if not state.hand_detected:
            return RenderCommand(
                event_type=RenderEventType.DRAW,
                event_position=(0.0, 0.0),
                trail=self.trail,
            )

        # 是 POINT 手勢 → 落筆,把食指尖加進軌跡
        if state.gesture == Gesture.POINT:
            self.trail.append(state.index_tip)
            # print(f"trail 長度: {len(self.trail)}")

        # 不是 POINT → 抬筆(什麼都不做,軌跡保留但不新增)
        return RenderCommand(
            event_type=RenderEventType.DRAW,
            event_position=state.index_tip,
            trail=self.trail,
        )