# Gesture Demo — 即時手勢互動系統

一個以攝影機即時辨識手勢、並觸發多種互動效果的系統。核心特色是**用機器學習模型取代手寫幾何規則來辨識手勢**,並透過乾淨的架構設計,讓這次替換「只改動一個模組、其餘完全不動」。系統支援一鍵切換「規則式」與「ML」兩種辨識器,可直接肉眼對比兩者差異。

---

## 這個專案在做什麼

攝影機捕捉手部 → 抽出 21 個關節點 → 辨識手勢 → 觸發對應互動(捏合觸發、抓取拖曳、手勢畫筆、手勢觸發圖片音效)。

整條管線是單向資料流,模組之間僅透過三個資料契約(dataclass)溝通,彼此不知道對方的內部實作:

```
camera → detector → smoother → recognizer → interaction → renderer
```

| 模組 | 職責 |
|---|---|
| `camera` | 讀取攝影機影格 |
| `detector` | 用 MediaPipe Hand Landmarker 抽出 21 個關節點,MediaPipe 只在此檔出現 |
| `smoother` | 對關節點做 EMA 平滑,減少抖動 |
| `recognizer` | 判斷手勢類別 + 計算幾何量測(捏合強度、指尖位置) |
| `interaction` | 策略模式,可切換多種互動邏輯 |
| `renderer` | 把手骨架、手勢、互動效果畫回畫面 |

資料契約(`contracts.py`):`HandLandmarks`、`GestureState`、`RenderCommand`。

---

## 架構:用 ML 替換規則式

專案原本用**手寫幾何規則**辨識手勢(判斷每根手指是否伸直)。這種做法對「手的角度」很敏感——例如比讚時,拇指必須大致垂直於畫面才判得準,手一旋轉就失效。

為了解決這個問題,辨識邏輯被替換成一個**自己訓練的手勢分類模型**。關鍵在於:

- **只改動 `recognizer.py` 內部**,上游的 `app`、`interaction`、`renderer` 一行都沒動。
- 因為所有模組都只依賴 `GestureState` 這個契約,不在乎它裡面的 `gesture` 欄位是「規則算出來的」還是「模型預測的」。
- 對外仍保持 `recognize(hands)` 的函式介面不變(內部改用有狀態的 class + 模組層級單例轉發),因此連 `app.py` 的呼叫方式都不需要改。

這驗證了契約隔離架構的價值:**核心邏輯可以整個抽換,系統其餘部分無感。**

```
                          +--------------------------+
                          |          app.py          |
                          +--------------------------+
                                       |  每一幀依序呼叫
                                       v
  +----------+   frame    +----------+  list[HandLandmarks]  +----------+
  |  camera  |----------->| detector |---------------------->| smoother |
  +----------+            +----------+                       +----------+
                                                                   |
                                                                   | list[HandLandmarks]
                                                                   v
  +----------+ RenderCommand +-------------+  GestureState  +------------+
  | renderer |<--------------| interaction |<---------------| recognizer |
  +----------+               +-------------+                +------------+
       |                            |                              |
       |                      1  trigger                    mode = "ml"   <- 預設
       |                      2  grab                               |
       |                      3  draw                               +-- features.normalize()
       |                      4  image_show                         +-- model.GestureMLP
       |                                                            +-- models/gesture_mlp.pth
       |
       |                                                     mode = "rules"
       |                                                            |
       |                                                            +-- is_finger_extended()
       v
    螢幕輸出
```

圖裡的英文名稱對應到的中文,以及幾個值得知道的細節:

- **`interaction` 的四種策略**按 `1` `2` `3` `4` 即時切換:`trigger` 捏合觸發、`grab` 抓取拖曳、`draw` 手勢畫筆、`image_show` 手勢圖片音效。四個都實作同一個 `Interaction` 介面,`app.py` 只持有一個 `self.interaction` 參考,切換就是換掉那個參考。
- **`recognizer` 的兩條路徑**按 `m` 切換,預設走 `ml`。兩條路徑對外都回傳同一個 `GestureState`,所以 `interaction` 和 `renderer` 完全不知道手勢是幾何規則算出來的還是模型預測的——這就是前面說的「核心邏輯整個抽換,系統其餘部分無感」。
- **`.pth` 載入失敗會自動退回 `rules` 模式**並印出提示。最常見的原因是新增手勢後類別數變了,舊的權重檔對不上 `len(GESTURE_LABELS)`,這時要重新錄資料並訓練。
- `detector` 是唯一碰 MediaPipe 的模組;`camera` 負責讀影格與水平翻轉;`smoother` 做 EMA 去抖動;`renderer` 畫骨架、手勢文字與互動視覺回饋。


### 雙模式切換

兩種辨識器可即時切換,用途是直接肉眼對比。實測差異最明顯的情境是「比讚並旋轉手腕」:規則式因為判斷拇指是否伸直而依賴手掌朝向,手一轉就失效;ML 模式因為訓練資料涵蓋了各種角度,仍然穩定。這也是當初決定改用模型的直接動機。

---

## 機器學習部分:完整流程

手勢辨識模型走過一遍完整的 ML 流程,從零開始:

### 1. 資料採集

- 重用系統既有的 `camera → detector`,寫一支採集腳本錄下帶標籤的關節點資料。
- 輸入特徵是 **21 個關節點座標**,而非原始影像。採集時連 z 一起存(63 欄),但訓練時只取 x, y(42 維)—— 實測 MediaPipe 的 z 是相對深度、雜訊大,加進去反而拖累準確率。原因:detector 已經完成從雜亂像素中定位手、抽出關節點這段最困難的工作;直接使用關節點,資料需求小、CPU 可訓練、且天生對背景與光線免疫。
- 錄製時遵守「控制變因」原則:每個手勢分數個 session,分別涵蓋位置、距離、角度的變化,並放慢動作以確保 detector 每一影格都能穩定定位。
- 每筆資料額外記錄 `session_id`,標記它來自哪一次連續錄製。

目前定義 15 種手勢:`fist`、`open`、`point`、`yeah`、`thumb_up`、`three`、`phone`、`ok`(最初的 8 類),以及 `four`、`seven`、`eight`、`gun`、`split`、`rock`、`middle`(後續新增)。15 類皆已錄製並完成訓練。


### 2. 正規化(讓模型對位置/大小免疫)

MediaPipe 輸出的 0~1 座標只消除了「螢幕解析度」這個變因,並未消除手在畫面的**位置**與**遠近**。正規化(`features.py`)進一步處理:

- **平移不變**:所有點減去手腕座標,讓手腕成為原點。
- **縮放不變**:所有點除以 palm_size(手腕到中指根的距離),消除遠近造成的尺度差異。

這一步是模型能對角度/位置 robust 的核心——同一個手勢不論在畫面何處、離鏡頭多遠,正規化後的特徵向量都幾乎一致。

### 3. 資料切分(避免 data leakage)

因為資料是連續影格,相鄰影格幾乎相同。若隨機切分 train/test,同一段動作的影格會同時落入兩邊,造成測試準確率虛高。

因此採用**分層 + 按 session 切分**(`session_split.py`):

- 對每個手勢各自切分,保證每個手勢在 train/test 都有代表(分層)。
- 以整個 session 為單位分配,同一段錄製不會被拆散(防洩漏)。

### 4. 模型與訓練

- 一個小型 MLP(42 → 128 → 64 → 類別數),以 ReLU 為激活函數,輸出層不接 softmax(交由 `CrossEntropyLoss` 內部處理)。輸出維度直接綁 `len(GESTURE_LABELS)`,新增手勢時不用手動改。模型定義獨立在 `model.py`,訓練腳本與辨識器共用同一份。
- 參數量僅數萬個,CPU 上即可快速訓練,無需 GPU。
- 訓練後將權重存為 `state_dict`,供辨識器載入。
- 超參數集中在 `train.py` 的 `CONFIG` 一個 dict 裡(`seed` / `epochs` / `batch_size` / `lr` / `test_ratio`),`train.py` 和 `train.ipynb` 共用這一份,兩邊不會各跑一套設定。

### 5. 可複現性(同樣的指令要跑出同樣的模型)

#### 問題:同樣的指令,跑兩次結果不一樣

原本 `py -m src.gesture_demo.train` 連跑兩次,test accuracy 會差大約 1 個百分點。這讓所有調整都無法判斷:補錄了資料、accuracy 升了 0.5%,是資料真的有幫助,還是這次剛好抽到比較好的初始權重?分不出來,就等於沒有辦法改進模型。

#### 原因:有三個隨機來源,原本只設了一個種子

| 隨機來源 | 在哪裡發生 | 原本狀況 |
|---|---|---|
| train/test 切分 | `session_split.py` 用 `np.random.default_rng(seed)` 挑哪些 session 當 test | 已有種子(42) |
| 模型權重初始化 | `nn.Linear` 的預設初始化,抽**全域** torch RNG | 沒設 → 每次起點都不同 |
| DataLoader 洗牌順序 | `shuffle=True` 決定每個 epoch 的 batch 順序,也抽**全域** torch RNG | 沒設 → 每次梯度路徑都不同 |

主因是**權重初始化**。神經網路從隨機權重開始,梯度下降只能找到附近的局部最小值,起點不同就會停在不同的地方。洗牌順序是第二個因素:batch 的順序決定了梯度更新的先後,同樣會把模型帶到不同的終點。

#### 做法:訓練開始前設定所有種子

`seeding.py` 把三個來源集中在一個地方管理:

```python
from src.gesture_demo.seeding import set_seed

set_seed(CONFIG["seed"])    # 必須在建立 model 和 DataLoader 之前
```

`set_seed()` 一次設好 `random`、`numpy`、`torch`、`torch.cuda` 和 `PYTHONHASHSEED`。這裡只用到 torch 和 numpy,其餘是為了以後加入資料增強之類的功能時不用再想起這件事。

#### 兩個容易踩的坑

**坑一:權重初始化和洗牌共用同一個全域 RNG。**

它們都從同一個 `torch` 全域 RNG 抽數字,所以會互相影響。假設你把模型從 `42 → 128 → 64` 改成 `42 → 256 → 64`,初始化要抽的隨機數變多了,於是輪到 DataLoader 抽的時候,拿到的是完全不同的數字——**洗牌順序也跟著變了**。結果 accuracy 的變化裡混進了洗牌的影響,你分不清是模型結構的功勞還是運氣。

解法是讓 DataLoader 用自己的 generator,不碰全域 RNG:

```python
train_loader = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    generator=make_loader_generator(config["seed"]),   # 獨立的 RNG
    worker_init_fn=seed_worker,
)
```

這樣改模型結構時,洗牌順序固定不動,兩個變因就分開了。

**坑二:notebook 的 cell 可以亂序重跑。**

`train.ipynb` 裡你可能只重跑「建模型」那個 cell 來重新開始訓練。如果種子只設在最上面的 cell,這次重跑就會拿到一組全新的隨機權重。所以「讀資料」和「建模型」兩個 cell 開頭都各自呼叫一次 `set_seed()`,單獨重跑任何一個都會得到一樣的結果。

另外 `train.ipynb` 直接 `from src.gesture_demo.train import CONFIG, build_loaders`,和 `train.py` 共用同一份設定。修改之前這兩邊是各寫一套的,notebook 跑 40 epochs、script 跑 30 epochs,本來就不可能對得起來。

#### 驗證

完整的 40 epochs 訓練,在乾淨的工作目錄上獨立跑兩次:

```
RUN 1: test accuracy = 0.9573    models/gesture_mlp.pth  sha256 4307c392...
RUN 2: test accuracy = 0.9573    models/gesture_mlp.pth  sha256 4307c392...
```

不只是準確率相同,連**產出的權重檔逐位元相同**。

對照組(把 `set_seed()` 拿掉,為了快速比較只跑 2 epochs):

```
RUN 1: loss = 0.2735952327    acc = 0.8936065775
RUN 2: loss = 0.2535955849    acc = 0.8837894220    ← 差約 1%
```

#### 種子還不夠:要記住三個「同樣」

種子保證的是這件事:

> 同樣的 code + 同樣的資料 + 同樣的設定 → 同樣的結果

但種子**不會幫你記住那三個「同樣」當時是什麼**。三個月後打開 `models/gesture_mlp.pth`,它只是 62KB 的數字,你看不出它是用哪一版 `gestures.csv`、哪一版 code、幾個 epoch 訓出來的。而這個專案的資料是會長大的(目前 33809 筆,每次補錄 session 就變),code 也還在改。

所以存模型時會在旁邊寫一份 `models/gesture_mlp.json`:

```json
{
  "created_at": "2026-09-11T12:50:47",
  "source": "train.py",
  "config": { "seed": 42, "epochs": 40, "batch_size": 64, "lr": 0.001, "test_ratio": 0.25 },
  "test_accuracy": 0.957295,
  "git_commit": "6848639",
  "data_csv": "data/raw/gestures.csv",
  "data_sha256": "e6052f4a8121abca0e0767d4cb0c4c23fe38cb7ec7fe6146fcb73687424a48eb",
  "gesture_labels": ["fist", "open", "..."],
  "torch_version": "2.13.0+cpu"
}
```

三個欄位各自回答一個問題:

| 欄位 | 回答什麼 | 怎麼產生 |
|---|---|---|
| `config` | 什麼設定?(seed、epochs、lr…) | 直接把 `CONFIG` 整份寫進去 |
| `git_commit` | 哪一版 code? | `git rev-parse --short HEAD`。原始碼有未 commit 的修改會標 `-dirty`,提醒你這次的結果對不回任何 commit |
| `data_sha256` | 哪一版資料? | 對 `gestures.csv` 算 SHA-256。檔名不會變但內容會長,只有雜湊認得出差異 |

**實際會用到的場景**:你補錄三個 session 重新訓練,accuracy 從 95.7% 升到 97%。比對新舊兩份 json——`data_sha256` 不同而 `git_commit` 相同,就確定是資料的功勞;如果兩個都變了,那這次實驗混了兩個變因,結論不能採信。

`git_commit` 的 dirty 判定會排除 `models/`,因為訓練本身就會寫出 `.pth` 和這份 `.json`,不排除的話每次訓練都把自己弄髒,`-dirty` 就永遠都在、失去意義。

> 這份訓練記錄是**可選的**。它不影響訓練結果,拿掉也不會破壞可複現性——刪掉 `save_run_metadata()`、`_file_sha256()`、`_git_commit()` 三個函式和對應的 `hashlib` / `json` / `subprocess` / `datetime` import 即可。核心是 `set_seed()` 和 `CONFIG`,那兩個不能拿掉。

#### 第三層:鎖住套件版本

`requirements.txt` 鎖定完整的版本清單,這也是可複現的一環——不同版本的 PyTorch 預設初始化方式可能不同,種子一樣也未必得到一樣的權重。

#### 檔案分工總表

| 檔案 | 在可複現性裡的角色 |
|---|---|
| `seeding.py` | 種子集中管理:`set_seed()`、`make_loader_generator()`、`seed_worker()` |
| `train.py` | `CONFIG`(唯一的超參數來源)、`build_loaders()`、寫出訓練記錄 |
| `train.ipynb` | import `train.py` 的 `CONFIG` 和 `build_loaders`,不自己另寫一套 |
| `session_split.py` | 切分自己收 `seed` 參數 |
| `requirements.txt` | 鎖住套件版本 |
| `models/gesture_mlp.json` | 每次訓練的來歷記錄(自動產生) |

### 6. 評估

- 以 held-out 的 test set 計算準確率與混淆矩陣。
- 目前 **test accuracy 95.73%**(15 類,8149 筆 test,`seed=42` / 40 epochs)。從 8 類擴到 15 類後準確率不降反升。
- 這個數字是固定種子後可複現的基準:同樣的 code + 資料 + 設定重跑,會得到完全一樣的結果。(加種子之前量到的 96.2% 只是無種子時代其中一次的成績,重跑不出來。)

---

## 已知限制

以誠實為原則,如實記錄目前的弱點:

以下數字取自當前的 `models/gesture_mlp.pth`(`seed=42`,可複現):

- **`thumb_up` / `phone` 互相吃掉**:`thumb_up` 的 recall 只有 0.75,其中 16% 被判成 `phone`;反向也有 5% 的 `phone` 被判成 `thumb_up`。這是目前最弱的一對(f1 各為 0.81 / 0.89)。
- **`fist` 有 13% 被判成 `middle`**,另有 5% 被判成 `thumb_up`,recall 掉到 0.81。`middle` 因此成為過度吸收的一方(precision 0.84,但 recall 1.00)。
- 較輕微的:`eight` 有 3% 被判成 `yeah`、`split` 有 4% 被判成 `rock`。
- 根本原因是這幾組的幾何差異只落在**拇指或小指單一根手指**的伸直狀態,在特徵空間中彼此接近;而拇指的姿態又特別容易受手掌朝向影響。
- 這較可能是**資料涵蓋不足**的問題(角度多樣性不夠),而非模型能力不足——其餘 11 個手勢的 f1 都在 0.97 以上(`open`、`point`、`three`、`ok` 達 1.00),顯示模型結構本身沒有問題。
- 已驗證這個判斷:`gun`、`three`、`middle` 原本 f1 分別只有 0.73 / 0.79 / 0.83(`gun` 有 23% 被判成 `eight`,因為中文比「八」跟 `gun` 幾乎是同一個手形,只差食指和中指有沒有張開),**各補錄一個角度更分散的 session、重新訓練後,f1 直接升到 0.97 / 1.00 / 0.99**,整體準確率也從 92.4% 拉到 96% 附近。剩下這幾組用同樣方法應該也能改善。
- 注意 `middle` 的 recall 1.00 / precision 0.84 是一組值得警惕的數字:它不是「判得準」,而是「把別人的也收進來」。單看 accuracy 會漏掉這件事,這也是要保留混淆矩陣的理由。

此外,模型每一影格獨立預測,類別偶爾會有跳動;目前尚未加入預測層級的時序平滑。

---

## 環境需求

- Python 3.12
- 攝影機
- 相依套件見 `requirements.txt`(主要為 PyTorch(CPU)、MediaPipe、OpenCV、pygame、NumPy、pandas、scikit-learn)
- `requirements.txt` 鎖定完整版本清單,是可複現訓練的一部分。檔案開頭的 `--extra-index-url https://download.pytorch.org/whl/cpu` 是必要的:`torch==2.13.0+cpu` 的 `+cpu` 是 local version,只存在 PyTorch 官方 index,PyPI 上找不到。

---

## 安裝與執行

```bash
# 建立並啟用虛擬環境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 安裝相依套件
pip install -r requirements.txt
```

需另外準備 MediaPipe 的手部模型檔 `hand_landmarker.task`,放在專案根目錄。

### 執行主程式

```bash
py -m src.gesture_demo.app
```

操作:

| 按鍵 | 功能 |
|---|---|
| `1` `2` `3` `4` | 切換互動模式(捏合觸發 / 抓取 / 畫筆 / 手勢圖片音效) |
| `m` | 切換辨識器(規則式 / ML) |
| `c` | 清除當前互動狀態(如清空畫筆) |
| `q` | 離開 |

---

## 自己訓練模型(選用)

專案已附上訓練好的模型(`models/gesture_mlp.pth`),可直接執行。若想自行收集資料並重新訓練:

```bash
# 1. 採集資料(按數字鍵選手勢、空白鍵開始/暫停、s 存檔、q 離開)
py -m src.gesture_demo.collectData.collect_data

# 2. 訓練並評估(會輸出 loss、test accuracy、混淆矩陣,並存出模型)
py -m src.gesture_demo.train
```

訓練會產出兩個檔案:`models/gesture_mlp.pth`(權重)和 `models/gesture_mlp.json`(這次訓練的設定、資料指紋、git commit、準確率)。

要調參就改 `train.py` 裡的 `CONFIG`,不要散在各處改——那份 dict 會整個寫進 json,事後才查得到某個模型是什麼設定訓出來的。固定 `seed` 的前提下重跑同樣的設定,結果會逐位元一致。

`train.ipynb` 是同一套流程的 notebook 版本,適合互動式調參;它直接 `from src.gesture_demo.train import CONFIG, build_loaders`,和 script 共用設定。

資料會存於 `data/`(未納入版控)。

---

## 專案結構

```
src/gesture_demo/
├── app.py              # 主迴圈,串接所有模組
├── camera.py           # 攝影機
├── detector.py         # MediaPipe 手部偵測(MediaPipe 唯一出現處)
├── smoother.py         # EMA 平滑
├── recognizer.py       # 手勢辨識(規則式 + ML 雙模式)+ 幾何量測
├── features.py         # 正規化
├── dataset.py          # PyTorch Dataset,資料前處理、GESTURE_LABELS
├── session_split.py    # 分層 + 按 session 切分
├── model.py            # 模型定義(GestureMLP)
├── seeding.py          # 隨機種子集中管理,訓練可複現
├── train.py            # 訓練、評估、存模型 + 訓練記錄
├── train.ipynb         # 同一套流程的 notebook 版本(共用 train.py 的 CONFIG)
├── contracts.py        # 資料契約(dataclass)
├── collectData/
│   └── collect_data.py # 資料採集腳本
└── interaction/        # 各種互動策略(策略模式)
    ├── base.py
    ├── trigger.py      # 捏合觸發
    ├── grab.py         # 抓取拖曳
    ├── draw.py         # 手勢畫筆
    └── image_show.py   # 手勢觸發圖片音效
```

> 註:`assets/`(互動用的圖片與音效)未納入版控。如需完整互動效果,請自行於 `assets/images/` 與 `assets/sounds/` 放入對應素材。
