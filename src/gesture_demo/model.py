import torch.nn as nn

from src.gesture_demo.dataset import NUM_CLASSES


# ── 模型 ──────────────────────────────────
class GestureMLP(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # 結構:42 (21 個點 × xy) → 128 → (ReLU) → 64 → (ReLU) → num_classes
        # 輸出維度直接綁 GESTURE_LABELS 的長度,加手勢時不用再手動改數字。
        # ⚠️ 類別數一變,舊的 models/gesture_mlp.pth 就不相容,必須重新訓練。
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
