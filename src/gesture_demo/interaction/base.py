from abc import ABC, abstractmethod
from src.gesture_demo.contracts import GestureState, RenderCommand


class Interaction(ABC):
    @abstractmethod
    def process(self, state: GestureState) -> RenderCommand:
        ...