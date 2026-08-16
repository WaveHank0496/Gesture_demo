from dataclasses import dataclass
from enum import Enum

# 手勢判斷
class Gesture(Enum):
    NONE = "none"
    FIST = "fist"
    OPEN = "open"
    POINT = "point"
    YEAH = "yeah"
    THUMB_UP = "thumb_up"
    THREE = "three"
    FOUR = "four"
    PHONE = "phone"

@dataclass
class GestureState:
    hand_detected: bool
    gesture: Gesture
    pinch_strength: float
    hand_position: tuple[float, float]
    index_tip: tuple[float, float]

# 關節點
@dataclass
class HandLandmarks:
    landmarks: list[tuple[float, float, float]]

# 事件觸發
class RenderEventType(Enum):
    NONE = "none"
    DRAW = "draw"
    GRAB = "grab"
    CLICK = "click"

@dataclass
class RenderCommand:
    event_type: RenderEventType
    event_position: tuple[float, float]
    trail: list[tuple[float, float]] = None 
    image_name: str = None