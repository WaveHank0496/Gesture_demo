from src.gesture_demo.contracts import HandLandmarks

class Smoother:
    def __init__(self, alpha: float= 0.5):
        self.alpha = alpha
        self.prev = None    # 上一幀的平滑結果

    def _smooth_one_hand(self, new_hand: HandLandmarks, prev_hand:HandLandmarks) -> HandLandmarks:
        smoothed_point = []
        for new_pt, prev_pt in zip(new_hand.landmarks, prev_hand.landmarks):
            x = self.alpha * new_pt[0] + (1 - self.alpha) * prev_pt[0]
            y = self.alpha * new_pt[1] + (1 - self.alpha) * prev_pt[1]
            z = self.alpha * new_pt[2] + (1 - self.alpha) * prev_pt[2]
            smoothed_point.append((x, y, z))
        return HandLandmarks(landmarks=smoothed_point)


    def smooth(self, hands: list[HandLandmarks]) -> list[HandLandmarks]:
        # 第一幀,或手數變動:沒有可對應的歷史,直接用這幀當結果
        if self.prev is None or len(self.prev) != len(hands):
            self.prev = hands
            return hands

        # 手數一致:逐隻手做 EMA 平滑
        result = []
        for new_hand, prev_hand in zip(hands, self.prev):
            result.append(self._smooth_one_hand(new_hand, prev_hand))

        self.prev = result      # 存起來給下一幀用
        return result


