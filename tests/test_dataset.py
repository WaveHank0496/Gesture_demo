import pandas as pd
from src.gesture_demo.dataset import GestureDataset, GESTURE_LABELS
from src.gesture_demo.session_split import session_train_test_split

cols = ["label", "session_id"] + [f"{ax}{i}" for i in range(21) for ax in ["x", "y", "z"]]
df = pd.read_csv("data/raw/gestures.csv", header=None, names=cols)

train_df, test_df = session_train_test_split(df, test_ratio=0.25, seed=42)

train_ds = GestureDataset(train_df)
test_ds = GestureDataset(test_df)

print("train 筆數:", len(train_ds))
print("test 筆數:", len(test_ds))

# 取第一筆看形狀
x, y = train_ds[0]
print("單筆 X shape:", x.shape, "(應該是 torch.Size([63]))")
print("單筆 y:", y, "(應該是 0~7 的整數)")
print("X dtype:", x.dtype, "(應該 float32)")
print("y dtype:", y.dtype, "(應該 int64/long)")

# 確認 normalize 生效:第一筆的手腕(前3個值)應該接近 0
print("手腕座標(前3個,應接近0):", x[:3])