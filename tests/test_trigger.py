from src.gesture_demo.interaction.trigger import PinchTrigger
from src.gesture_demo.contracts import GestureState, Gesture, RenderEventType


def fake_state(pinch):
    # 假造一個 GestureState,只有 pinch_strength 重要,其他填合理值
    return GestureState(
        hand_detected=True,
        gesture=Gesture.NONE,
        pinch_strength=pinch,
        hand_position=(0.5, 0.5),
        index_tip=(0.5, 0.5),
    )


def test_pinch_edge_detection():
    trigger = PinchTrigger()

    # 沒捏 → NONE
    assert trigger.process(fake_state(0.1)).event_type == RenderEventType.NONE
    # 剛捏下去 → CLICK(邊緣!)
    assert trigger.process(fake_state(0.9)).event_type == RenderEventType.CLICK
    # 還在捏 → NONE(不重複觸發,這是關鍵)
    assert trigger.process(fake_state(0.9)).event_type == RenderEventType.NONE
    # 放開 → NONE
    assert trigger.process(fake_state(0.1)).event_type == RenderEventType.NONE
    # 再捏下去 → 又 CLICK(能重複觸發)
    assert trigger.process(fake_state(0.9)).event_type == RenderEventType.CLICK

    print("test_pinch_edge_detection 通過")


if __name__ == "__main__":
    test_pinch_edge_detection()
    print("全部測試通過")