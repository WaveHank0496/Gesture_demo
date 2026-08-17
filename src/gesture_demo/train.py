import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.gesture_demo.dataset import GestureDataset
from src.gesture_demo.session_split import session_train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from src.gesture_demo.dataset import GESTURE_LABELS


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


# ── 主程式 ────────────────────────────────
def main():
    # 讀資料 + 切分(照抄)
    cols = ["label", "session_id"] + [f"{ax}{i}" for i in range(21) for ax in ["x", "y", "z"]]
    df = pd.read_csv("data/raw/gestures.csv", header=None, names=cols)
    train_df, test_df = session_train_test_split(df, test_ratio=0.25, seed=42)

    train_ds = GestureDataset(train_df)
    test_ds = GestureDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 建立模型、loss、optimizer(照抄)
    model = GestureMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 訓練(照抄迴圈,五步你填)
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            # TODO 你填:訓練五步
            # 1. 梯度歸零
            optimizer.zero_grad()
            # 2. 前向:pred = model(X_batch)
            pred = model(X_batch)
            # 3. 算 loss:criterion(pred, y_batch)
            loss = criterion(pred, y_batch)
            # 4. 反向:loss.backward()
            loss.backward()
            # 5. 更新:optimizer.step()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch+1:2d}: loss = {avg_loss:.4f}")


    # ── 評估(訓練後)──────────────────
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():                    # 評估不需要梯度
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)            # (batch, 8) 的 logits
            pred_labels = pred.argmax(dim=1)  # 取每列分數最高的那類 → 預測類別

            correct += (pred_labels == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(pred_labels.tolist())
            all_labels.extend(y_batch.tolist())

    print(f"\ntest accuracy: {correct/total:.4f}")

    print("\n混淆矩陣(列=真實, 行=預測):")
    cm = confusion_matrix(all_labels, all_preds)
    # 印表頭
    print("真實\\預測  " + " ".join(f"{name[:5]:>6}" for name in GESTURE_LABELS))
    for i, name in enumerate(GESTURE_LABELS):
        row = " ".join(f"{cm[i][j]:>6}" for j in range(len(GESTURE_LABELS)))
        print(f"{name[:8]:>8}  {row}")

    print("\n每類詳細指標:")
    print(classification_report(all_labels, all_preds, target_names=GESTURE_LABELS))

    # ── 存模型 ──────────────────
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/gesture_mlp.pth")
    print("\n模型已存到 models/gesture_mlp.pth")


if __name__ == "__main__":
    main()