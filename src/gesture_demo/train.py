import hashlib
import json
import os
import subprocess
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.gesture_demo.dataset import GestureDataset
from src.gesture_demo.session_split import session_train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from src.gesture_demo.dataset import GESTURE_LABELS
from src.gesture_demo.model import GestureMLP
from src.gesture_demo.seeding import SEED, make_loader_generator, seed_worker, set_seed


# ── 超參數(單一來源,notebook 也 import 這份)────────────
# 想調參就只改這裡。每個值都會寫進 models/gesture_mlp.json,
# 事後才查得到「這個 .pth 是用什麼設定訓練出來的」。
CONFIG = {
    "seed": SEED,
    "epochs": 40,
    "batch_size": 64,
    "lr": 0.001,
    "test_ratio": 0.25,
}

DATA_CSV = "data/raw/gestures.csv"
MODEL_PATH = "models/gesture_mlp.pth"


def _file_sha256(path: str) -> str:
    """資料檔的指紋。csv 一直在長,沒有這個就分不出模型是用哪一版資料訓的。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """記下 code 的版本。有未 commit 的修改會標 -dirty。"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "unknown"


def build_loaders(config: dict = CONFIG) -> tuple[DataLoader, DataLoader]:
    """讀資料 + 切分 + 包成 DataLoader。notebook 直接呼叫這個,兩邊不會走鐘。"""
    cols = ["label", "session_id"] + [f"{ax}{i}" for i in range(21) for ax in ["x", "y", "z"]]
    df = pd.read_csv(DATA_CSV, header=None, names=cols)
    train_df, test_df = session_train_test_split(
        df, test_ratio=config["test_ratio"], seed=config["seed"]
    )

    train_ds = GestureDataset(train_df)
    test_ds = GestureDataset(test_df)

    # generator 是洗牌可複現的關鍵;shuffle=True 預設抽全域 RNG,
    # 那個 RNG 已經被模型初始化消耗過,順序會隨模型結構飄。
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=make_loader_generator(config["seed"]),
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    return train_loader, test_loader


# ── 主程式 ────────────────────────────────
def main():
    # 所有隨機來源的種子。一定要在建 model / DataLoader 之前。
    set_seed(CONFIG["seed"])

    train_loader, test_loader = build_loaders(CONFIG)

    # 建立模型、loss、optimizer(照抄)
    model = GestureMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # 訓練(照抄迴圈,五步你填)
    for epoch in range(CONFIG["epochs"]):
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
            pred = model(X_batch)            # (batch, 類別數) 的 logits
            pred_labels = pred.argmax(dim=1)  # 取每列分數最高的那類 → 預測類別

            correct += (pred_labels == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(pred_labels.tolist())
            all_labels.extend(y_batch.tolist())

    accuracy = correct / total
    print(f"\ntest accuracy: {accuracy:.4f}")

    print("\n混淆矩陣(列=真實, 行=預測):")
    # 明確指定 labels:某一類還沒資料 / 沒被切進 test 時,矩陣才不會少一列一行
    all_idx = list(range(len(GESTURE_LABELS)))
    cm = confusion_matrix(all_labels, all_preds, labels=all_idx)
    # 印表頭
    print("真實\預測  " + " ".join(f"{name[:5]:>6}" for name in GESTURE_LABELS))
    for i, name in enumerate(GESTURE_LABELS):
        row = " ".join(f"{cm[i][j]:>6}" for j in range(len(GESTURE_LABELS)))
        print(f"{name[:8]:>8}  {row}")

    print("\n每類詳細指標:")
    print(classification_report(all_labels, all_preds,
                                labels=all_idx, target_names=GESTURE_LABELS,
                                zero_division=0))

    # ── 存模型 ──────────────────
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    save_run_metadata(MODEL_PATH, CONFIG, accuracy, source="train.py")
    print(f"\n模型已存到 {MODEL_PATH}")


def save_run_metadata(model_path: str, config: dict, accuracy: float, source: str) -> None:
    """把「這次訓練是怎麼跑的」寫在模型旁邊,檔名同 .pth 但副檔名 .json。

    種子只保證「同樣的 code + 同樣的資料 + 同樣的設定」會跑出同樣的結果;
    這份 json 負責記住後面那三個「同樣」到底是什麼。
    """
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "config": config,
        "test_accuracy": round(accuracy, 6),
        "git_commit": _git_commit(),
        "data_csv": DATA_CSV,
        "data_sha256": _file_sha256(DATA_CSV),
        "gesture_labels": list(GESTURE_LABELS),
        "torch_version": torch.__version__,
    }
    meta_path = os.path.splitext(model_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"訓練記錄已存到 {meta_path}")


if __name__ == "__main__":
    main()
