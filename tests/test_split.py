import pandas as pd
from src.gesture_demo.session_split import session_train_test_split, report_split_summary

cols = ["label", "session_id"] + [f"{ax}{i}" for i in range(21) for ax in ["x", "y", "z"]]
df = pd.read_csv("data/raw/gestures.csv", header=None, names=cols)

train_df, test_df = session_train_test_split(df, test_ratio=0.2, seed=42)
report_split_summary(train_df, test_df)