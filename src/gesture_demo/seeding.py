"""集中管理隨機種子,讓每一次訓練都可以複現。

訓練裡的隨機來源有三個,漏掉任何一個,同樣的指令就會跑出不同的模型:
  1. train/test 切分     → session_split.py 自己收 seed 參數
  2. 模型權重初始化       → nn.Linear 抽「全域 torch RNG」
  3. DataLoader 洗牌順序  → shuffle=True 也抽「全域 torch RNG」

第 2、3 項共用同一個全域 RNG,所以它們會互相影響:改了模型結構,
抽掉的隨機數個數就變了,洗牌順序也跟著變。為了把兩者解耦,
DataLoader 一律用 make_loader_generator() 給的獨立 generator。
"""

import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """設定所有全域 RNG。訓練開始前呼叫一次。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # 沒有 GPU 時這行是 no-op,不會報錯

    if torch.cuda.is_available():
        # cuDNN 會自動挑「最快」的卷積演算法,同樣輸入可能選到不同演算法 → 結果有微小差異。
        # 這個 MLP 沒有卷積層,留著只是為了以後換模型時不用再想起這件事。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader_generator(seed: int = SEED) -> torch.Generator:
    """給 DataLoader 用的獨立 generator,讓洗牌順序不受模型初始化影響。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader num_workers > 0 時的 worker_init_fn。

    目前 num_workers=0(單一 process)用不到,但一旦改成多 worker,
    每個 worker 會拿到不同的 base seed,不設就又不可複現了。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
