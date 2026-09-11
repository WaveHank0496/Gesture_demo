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
                    ┌─────────────────────────────────────────────┐
                    │                   App (主程式)              │
                    │        持有所有模組、驅動主迴圈、處理按鍵      │
                    └─────────────────────────────────────────────┘
                                        │
    每一幀影像 (frame)                    │  依序呼叫
                                        ▼
  ┌──────────┐   frame    ┌──────────┐  list[HandLandmarks]  ┌──────────┐
  │  camera  │──────────▶│ detector │──────────────────────▶│ smoother │
  │ 攝影機來源│            │ 手部偵測  │                       │ 平滑濾波 │
  └──────────┘            └──────────┘                       └──────────┘
   讀取影像                 MediaPipe                          EMA 去抖動
   水平翻轉                 (唯一碰 MediaPipe 的模組)                  │
                                                                    │ list[HandLandmarks]
                                                                    ▼
  ┌──────────┐  RenderCommand  ┌─────────────┐  GestureState  ┌────────────┐
  │ renderer │◀───────────────│ interaction │◀────────────── │ recognizer │
  │  渲染輸出 │                 │  互動策略    │                │  手勢辨識  │
  └──────────┘                 └─────────────┘                └────────────┘
   畫骨架 / 手勢文字             策略模式：三種可插拔              幾何規則辨識手勢
   互動視覺回饋                 Trigger / Grab / Draw            計算捏合程度
        │
        ▼
     螢幕輸出
```


### 雙模式切換

執行時按 `m` 可即時切換「規則式」與「ML」兩種辨識器,方便直接對比。實測在「比讚並旋轉手腕」這類情境下,ML 模式明顯比規則式穩定。

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

訓練裡有三個獨立的隨機來源,漏掉任何一個,同樣的指令就會產出不同的模型:

| 隨機來源 | 位置 |
|---|---|
| train/test 切分 | `session_split.py`,收 `seed` 參數 |
| 模型權重初始化 | `nn.Linear` 的預設初始化,抽**全域** torch RNG |
| DataLoader 洗牌順序 | `shuffle=True`,同樣抽**全域** torch RNG |

`seeding.py` 把三者集中管理,訓練開始前呼叫一次 `set_seed()`(必須在建立 model 和 DataLoader **之前**)。

一個容易漏掉的細節:後兩項共用同一個全域 RNG,所以它們會互相影響——只要改了模型結構(例如 128 → 256),抽掉的隨機數個數就變了,洗牌順序也跟著飄,於是分不清準確率的變化是來自模型還是來自洗牌。因此 DataLoader 一律吃 `make_loader_generator()` 給的**獨立** generator,把兩者解耦。

Notebook 另有一個坑:cell 可以亂序重跑。所以「讀資料」與「建模型」兩個 cell 開頭都各自重設一次種子,單獨重跑其中一個也會得到一樣的結果。

實測(同一份 code 連跑兩次):

```
修好後   RUN 1: loss = 0.2578029254   acc = 0.8938520064
         RUN 2: loss = 0.2578029254   acc = 0.8938520064   ← 逐位元一致

沒種子時 RUN 1: loss = 0.2735952327   acc = 0.8936065775
         RUN 2: loss = 0.2535955849   acc = 0.8837894220   ← 差約 1%
```

**種子只保證「同樣的 code + 同樣的資料 + 同樣的設定」跑出同樣結果**,它不會幫你記住那三個「同樣」當時是什麼。`gestures.csv` 會隨著補錄資料一直長,`.pth` 檔本身看不出是用哪一版資料、哪個 commit 訓的。因此每次存模型時,會在旁邊寫一份同名的 `models/gesture_mlp.json`(以下為格式範例):

```json
{
  "created_at": "2026-09-11T12:28:01",
  "source": "train.py",
  "config": { "seed": 42, "epochs": 40, "batch_size": 64, "lr": 0.001, "test_ratio": 0.25 },
  "test_accuracy": 0.962,
  "git_commit": "31ccf0e",
  "data_sha256": "e6052f4a8121...",
  "gesture_labels": ["fist", "open", "..."],
  "torch_version": "2.13.0+cpu"
}
```

`data_sha256` 認資料版本、`git_commit` 認 code 版本(有未 commit 的修改會標 `-dirty`)、`config` 認設定。三個對得上,就複現得出來。

同理,`requirements.txt` 鎖完整版本清單也是複現的一環——不同版本的 PyTorch 預設初始化方式可能不同,種子一樣也未必得到一樣的權重。

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
- 已驗證這個判斷:`gun`、`three`、`middle` 原本 f1 分別只有 0.73 / 0.79 / 0.83(`gun` 有 23% 被判成 `eight`,因為中文比「八」跟 `gun` 幾乎是同一個手形,只差手腕角度),**各補錄一個角度更分散的 session、重新訓練後,f1 直接升到 0.97 / 1.00 / 0.99**,整體準確率也從 92.4% 拉到 96% 附近。剩下這幾組用同樣方法應該也能改善。
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
