"""手勢設定的一致性檢查。

新增手勢時最容易出的錯就是「有一個地方漏改」。這支測試把四個地方綁在一起:
GESTURE_LABELS(類別編號)、Gesture enum(辨識器轉回手勢)、
LABEL_MAP(錄資料的按鍵)、模型輸出維度。
"""
import torch

from src.gesture_demo.contracts import Gesture
from src.gesture_demo.dataset import GESTURE_LABELS, NUM_CLASSES
from src.gesture_demo.model import GestureMLP
from src.gesture_demo.collectData.collect_data import LABEL_MAP, TRAINED_LABEL_MAP, NEW_LABEL_MAP


def test_labels_have_enum_member():
    # 辨識器會做 Gesture(GESTURE_LABELS[idx]),enum 少一個就會 ValueError
    enum_values = {g.value for g in Gesture}
    missing = [name for name in GESTURE_LABELS if name not in enum_values]
    assert not missing, f"contracts.py 的 Gesture enum 缺少:{missing}"
    print("test_labels_have_enum_member 通過")


def test_labels_are_unique():
    assert len(GESTURE_LABELS) == len(set(GESTURE_LABELS)), "GESTURE_LABELS 有重複"
    print("test_labels_are_unique 通過")


def test_every_label_is_recordable():
    # 每個要訓練的手勢都要有一個按鍵能錄到,否則永遠收不到資料
    mapped = set(LABEL_MAP.values())
    missing = [name for name in GESTURE_LABELS if name not in mapped]
    assert not missing, f"collect_data.py 的 LABEL_MAP 缺少按鍵:{missing}"
    print("test_every_label_is_recordable 通過")


def test_no_key_collision():
    # 兩張表不能撞鍵,也不能撞到已定義的功能鍵
    overlap = set(TRAINED_LABEL_MAP) & set(NEW_LABEL_MAP)
    assert not overlap, f"按鍵重複定義:{overlap}"
    reserved = {'q', 's', ' '}          # q=離開, s=存檔, 空白=開始/暫停
    clash = set(LABEL_MAP) & reserved
    assert not clash, f"按鍵跟功能鍵衝突:{clash}"
    print("test_no_key_collision 通過")


def test_model_output_matches_labels():
    out = GestureMLP()(torch.zeros(1, 42))
    assert out.shape[1] == NUM_CLASSES == len(GESTURE_LABELS), (
        f"模型輸出 {out.shape[1]} 類,GESTURE_LABELS 有 {len(GESTURE_LABELS)} 類"
    )
    print("test_model_output_matches_labels 通過")


if __name__ == "__main__":
    test_labels_have_enum_member()
    test_labels_are_unique()
    test_every_label_is_recordable()
    test_no_key_collision()
    test_model_output_matches_labels()
    print("全部測試通過")
