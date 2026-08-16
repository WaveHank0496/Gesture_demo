# Gesture Demo — 即時手勢互動系統

一個以 **MediaPipe** 為基礎的即時手部互動系統：偵測手部關節點、辨識手勢、並提供三種可即時切換的互動模式（捏合觸發、抓取拖動、空中畫筆）。純 CPU 即可運行，不需 GPU。

這個專案的重點**不在功能數量，而在架構設計**。整套系統以「單向資料管線 + 明確資料契約」為核心，做到六個模組彼此解耦、互動策略可插拔、第三方函式庫被隔離在單一模組。以下文件會著重說明這些設計決策背後的理由。

---

## Demo 一覽

| 模式 | 觸發手勢 | 效果 |
|------|---------|------|
| 捏合觸發 (Trigger) | 拇指 + 食指捏合 | 在捏合位置出現一次點擊回饋（紅圈） |
| 抓取拖動 (Grab) | 捏住畫面上的方塊並移動 | 方塊跟隨手移動，放開即固定 |
| 空中畫筆 (Draw) | 食指指向 (POINT) | 指尖經過的軌跡連成線，多筆獨立 |

**操作按鍵**：`1` / `2` / `3` 切換三種互動 · `c` 清空畫布 · 張開手掌亦可清空畫布 · `q` 離開

---

## 系統架構

整個系統是一條 **單向資料管線 (unidirectional pipeline)**：每一幀影像進來，依序經過六個模組加工，最後輸出到畫面。

```
                    ┌─────────────────────────────────────────────┐
                    │                   App (主程式)                │
                    │        持有所有模組、驅動主迴圈、處理按鍵         │
                    └─────────────────────────────────────────────┘
                                        │
    每一幀影像 (frame)                    │  依序呼叫
                                        ▼
  ┌──────────┐   frame    ┌──────────┐  list[HandLandmarks]  ┌──────────┐
  │  camera  │──────────▶│ detector │──────────────────────▶│ smoother │
  │ 攝影機來源 │            │ 手部偵測  │                        │  平滑濾波 │
  └──────────┘            └──────────┘                        └──────────┘
   讀取影像                 MediaPipe                          EMA 去抖動
   水平翻轉                 (唯一碰 MediaPipe 的模組)                  │
                                                                    │ list[HandLandmarks]
                                                                    ▼
  ┌──────────┐  RenderCommand  ┌─────────────┐  GestureState  ┌────────────┐
  │ renderer │◀───────────────│ interaction │◀──────────────│ recognizer │
  │  渲染輸出 │                 │  互動策略    │                │  手勢辨識   │
  └──────────┘                 └─────────────┘                └────────────┘
   畫骨架 / 手勢文字             策略模式：三種可插拔              幾何規則辨識手勢
   互動視覺回饋                 Trigger / Grab / Draw            計算捏合程度
        │
        ▼
     螢幕輸出
```

模組之間傳遞的不是彼此的內部物件，而是三個定義好的**資料契約 (data contracts)**：

- **`HandLandmarks`** — detector 的輸出，代表「一隻手」的 21 個關節點座標。多隻手用 `list[HandLandmarks]` 表達。
- **`GestureState`** — recognizer 的輸出，系統的「語意層」：目前手勢、捏合程度、關鍵座標。
- **`RenderCommand`** — interaction 的輸出，描述「發生了什麼事件」，供 renderer 繪製。

---

## 關鍵設計決策

這一節是這份文件的重點——說明每個決策的**理由**，而非只描述「做了什麼」。

### 1. 單向管線 + 資料契約：讓模組解耦

管線上每一段只依賴前一段的「輸出資料格式」，不依賴它「如何算出來」。detector 吐出 `HandLandmarks`，下游只認得這個格式，完全不需要知道它是 MediaPipe 算的、還是別的模型。

**帶來的好處**：任何一段都能獨立替換與測試。例如可以手動塞一個假的 `GestureState` 給互動模組做單元測試，完全不需開攝影機。

### 2. 依賴隔離：MediaPipe 只出現在 detector

整個專案只有 `detector.py` 一個檔案接觸 MediaPipe。這讓第三方函式庫的變動被限縮在單一模組。

**實際驗證**：開發過程中，MediaPipe 從 legacy Solutions API 換成新版 Tasks API（因套件升級到 1.0 移除了舊 API），改動只發生在 `detector.py`，其他模組一行未動。

### 3. 策略模式：三種互動可插拔

三種互動（Trigger / Grab / Draw）輸入輸出完全相同（收 `GestureState`、吐 `RenderCommand`），只有中間邏輯不同。透過抽象基底類別 `Interaction` 定義共同介面，各互動實作它，主程式想掛哪個就掛哪個。

**實際驗證**：新增第二、第三種互動時，`app.py` 只改兩行（import + 建立物件），detector / smoother / recognizer / renderer 全部不動。切換互動時，呼叫端 `self.interaction.process(state)` 也完全不變。

### 4. 向後相容的契約演進

畫筆功能需要 `RenderCommand` 攜帶「一整串軌跡」，但原本它只有單一座標。解法是**新增一個有預設值的欄位** `trail=None`：新功能用得到它，舊的 Trigger / Grab 不傳它也照常運作。這是在不破壞現有功能下擴充契約的標準做法。

### 5. EMA 平滑：穩定與延遲的權衡

MediaPipe 逐幀獨立推論會產生高頻抖動。平滑模組使用**指數移動平均 (EMA)**：`平滑值 = α × 新值 + (1−α) × 前一次平滑值`，時間與空間複雜度皆為 **O(1)**（只需保留前一次結果）。

單一參數 `α` 控制核心權衡：α 大 → 跟手但仍抖；α 小 → 穩定但延遲高。這是即時系統無法迴避的取捨，沒有免費的午餐。

### 6. 邊緣偵測：狀態轉變而非狀態本身

捏合觸發若寫成「捏合程度 > 門檻就觸發」，捏著不放時會每幀狂觸發。正確做法是偵測「從沒捏 → 捏下去」的**瞬間**（本幀捏合 + 上幀未捏合），因此每個互動需要記住上一幀的狀態。畫筆的「開新筆」也用同一模式。

---

## 專案結構

```
Gesture_demo/
├── src/
│   └── gesture_demo/
│       ├── contracts.py        # 三個資料契約 + Gesture / RenderEventType 列舉
│       ├── camera.py           # 攝影機來源 (含水平翻轉)
│       ├── detector.py         # MediaPipe 手部偵測 (唯一碰 MediaPipe 的模組)
│       ├── smoother.py         # EMA 平滑濾波
│       ├── recognizer.py       # 幾何規則辨識手勢 + 捏合程度
│       ├── renderer.py         # 渲染骨架、手勢文字、互動視覺回饋
│       ├── app.py              # 主程式：組裝六個模組、驅動主迴圈
│       └── interaction/        # 互動策略 (策略模式)
│           ├── base.py         #   Interaction 抽象基底類別 (共同介面)
│           ├── trigger.py      #   捏合觸發
│           ├── grab.py         #   抓取拖動
│           └── draw.py         #   空中畫筆
├── tests/                      # 單元測試
│   ├── test_recognizer.py      #   distance() 幾何函式
│   └── test_trigger.py         #   捏合觸發的邊緣偵測邏輯
├── requirements.txt
└── README.md
```

---

## 環境需求與安裝

- **Python 3.12**（MediaPipe 目前不支援 3.13）
- 一個可用的攝影機
- 純 CPU 即可運行

```bash
# 建立虛擬環境 (Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 安裝依賴
pip install -r requirements.txt

# 下載 MediaPipe 手部模型
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## 執行

一律從**專案根目錄**、以模組方式執行（確保 import 路徑一致）：

```bash
python -m src.gesture_demo.app
```

執行後對著鏡頭比手，用 `1` / `2` / `3` 切換互動、`c` 清空、`q` 離開。

### 執行測試

```bash
python -m tests.test_recognizer
python -m tests.test_trigger
```

測試完全以假資料驗證純邏輯（例如餵一連串假的 `GestureState` 給捏合觸發，斷言「捏一次只觸發一次、捏著不重複、放開再捏能再觸發」），不需開啟攝影機。

---

## 已知限制與未來擴充

以下為刻意保留的擴充空間——優先做完核心的穩定與完整，而非堆疊功能。

- **多手互動**：偵測層已支援多手（可畫多隻手骨架），但語意/互動層目前聚焦單手。多手互動需要穩定的手部追蹤（最近鄰匹配），可於未來加入。
- **多手平滑**：目前手數變動的那一幀會重置平滑。完整版需以手部追蹤將前後幀的手一一對應。
- **更多手勢**：目前用幾何規則辨識（伸指組合 + 捏合距離）。可繼續加規則，或改接 MediaPipe Gesture Recognizer / 自訓模型——因辨識邏輯隔離在 `recognizer.py`，替換不影響其他模組。
- **更穩健的手勢判斷**：目前的伸指判斷對手部大幅旋轉較敏感，未來可改用向量夾角。
- **特效層**：`RenderCommand` 已設計為「描述事件」而非「執行繪製」，未來可新增 effects 模組訂閱同一份 `RenderCommand` 做進階視覺，互動邏輯無需改動。
- **當前模式提示**：畫面可加上「目前互動模式」的文字提示，改善展示體驗。

---

## 技術棧

Python 3.12 · MediaPipe (Tasks API) · OpenCV · dataclasses / Enum / ABC