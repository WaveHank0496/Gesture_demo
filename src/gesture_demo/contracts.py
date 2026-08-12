from dataclasses import dataclass
from enum import Enum

class Gesture(Enum):
    NONE = "none"
    FIST = "fist"
    OPEN = "open"
    POINT = "point"

@dataclass
class GestureState:
    hand_detected: bool
    gesture: Gesture
    pinch_strength: float
    hand_position: tuple[float, float]
    index_tip: tuple[float, float]