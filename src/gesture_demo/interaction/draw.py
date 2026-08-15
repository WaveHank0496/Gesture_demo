from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType, Gesture


class DrawPen(Interaction):
    def __init__(self):
        self.strokes = []
        self.was_drawing = False    # 上一幀是否在畫


    def process(self, state: GestureState) -> RenderCommand:
        is_drawing = state.hand_detected and state.gesture == Gesture.POINT
        
        from src.gesture_demo.interaction.base import Interaction
from src.gesture_demo.contracts import GestureState, RenderCommand, RenderEventType, Gesture


class DrawPen(Interaction):
    def __init__(self):
        self.strokes = []            # 所有筆跡:list of (list of points)
        self.was_drawing = False     # 上一幀是否在畫(邊緣偵測用)

    def process(self, state: GestureState) -> RenderCommand:
        is_drawing = state.hand_detected and state.gesture == Gesture.POINT

        if is_drawing:
            if not self.was_drawing:
                # 邊緣:從沒畫→開始畫,開一筆新的
                self.strokes.append([])
            # 落筆中:把點加進「當前這一筆」(最後一筆)
            self.strokes[-1].append(state.index_tip)

        self.was_drawing = is_drawing      # 更新狀態(邊緣偵測必備)

        return RenderCommand(
            event_type=RenderEventType.DRAW,
            event_position=state.index_tip if state.hand_detected else (0.0, 0.0),
            trail=self.strokes,       # 現在傳的是「多筆」
        )