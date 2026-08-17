import numpy as np
import pandas as pd


def session_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    分層 + 按 session 切分 train/test。
    對「每個 label 各自」獨立切分它的 session,保證:
      1. 每個手勢在 train 和 test 都有代表(分層,stratified)
      2. 同一 session 不會被拆到兩邊(防 data leakage)

    前提:df 有 'label' 和 'session_id' 兩欄。
    注意:每個手勢 session 數少(如 4 個),test_ratio=0.25 代表
    每個手勢拿 1 個 session 當 test。test set 多樣性受限於此,
    是資料結構的先天限制。
    """
    if "session_id" not in df.columns:
        raise ValueError("df 缺少 session_id 欄位,無法按 session 切分")

    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for label, group in df.groupby("label"):
        sessions = group["session_id"].unique()
        rng.shuffle(sessions)

        n_test = max(1, int(len(sessions) * test_ratio))
        test_sessions = set(sessions[:n_test])

        is_test = group["session_id"].isin(test_sessions)
        train_parts.append(group[~is_test])
        test_parts.append(group[is_test])

    train_df = pd.concat(train_parts).copy()
    test_df = pd.concat(test_parts).copy()

    return train_df, test_df


def report_split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """印出切分後的檢查:每個 label 在 train/test 是否都有涵蓋、比例合不合理。"""
    total = len(train_df) + len(test_df)
    print(f"train: {len(train_df)} 筆 ({len(train_df)/total:.1%})")
    print(f"test:  {len(test_df)} 筆 ({len(test_df)/total:.1%})")
    print()

    train_counts = train_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()
    summary = pd.DataFrame({"train": train_counts, "test": test_counts}).fillna(0).astype(int)
    print(summary)

    missing_in_test = summary[summary["test"] == 0]
    if not missing_in_test.empty:
        print()
        print("警告:以下 label 在 test set 裡完全沒有出現:")
        print(missing_in_test.index.tolist())
    else:
        print()
        print("每個 label 在 train/test 都有涵蓋。")