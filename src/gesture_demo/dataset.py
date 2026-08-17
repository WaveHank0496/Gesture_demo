import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.gesture_demo.features import normalize


# label 字串 <-> 整數索引的對應。模型輸出的是整數類別,要有固定映射。
GESTURE_LABELS = ["fist", "open", "point", "yeah", "thumb_up", "three", "phone", "ok"]
LABEL_TO_IDX = {name: i for i, name in enumerate(GESTURE_LABELS)}


class GestureDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        df: 有 label, session_id, 和 63 個座標欄的 DataFrame。
        建立時一次性 normalize 好所有資料。
        """
        self.X = []   # 每筆是 normalize 後攤平的 63 維向量
        self.y = []   # 每筆是整數 label

        for _, row in df.iterrows():
            # --- 你要填的部分 START ---
            # 1. 從 row 取出 21 個 (x,y,z) tuple,組成 list[tuple]
            #    提示:欄位順序是 x0,y0,z0, x1,y1,z1, ...
            landmarks = []
            for i in range(21):
                # landmarks.append((row[f"x{i}"], row[f"y{i}"], row[f"z{i}"])) 後來發現 z 座標在搞
                landmarks.append((row[f"x{i}"], row[f"y{i}"]))

            # 2. 呼叫 normalize() 得到正規化後的 21 個點
            normalized = normalize(landmarks)

            # 3. 把 21 個點攤平成一個長度 63 的 list/array
            flat = []
            for (x, y) in normalized:
                flat.extend([x, y])

            # 4. 把字串 label 轉成整數索引
            label_idx = LABEL_TO_IDX[row["label"]]

            self.X.append(flat)
            self.y.append(label_idx)

        # 轉成 tensor:X 是 float32,y 是 long(分類 label 的慣例)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]