import torch.nn as nn


# ── 模型 ──────────────────────────────────
class GestureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 你填:定義三層 Linear + ReLU
        # 結構:63 → 128 → (ReLU) → 64 → (ReLU) → 8
        # 用 nn.Linear(輸入, 輸出) 和 nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.net(x)
