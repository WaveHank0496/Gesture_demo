import numpy as np


def euclidean_distance_3d(p1: tuple[float, float, float],
                           p2: tuple[float, float, float]) -> float:
    """算兩個 3D 點的歐氏距離。用於量測 palm_size(手腕到中指根的骨長)。"""
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def normalize(landmarks: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """
    把 21 個原始關節點,轉成「平移不變 + 縮放不變」的正規化座標。
    - 平移:所有點減去手腕(landmarks[0]),讓手腕變成原點。
    - 縮放:所有點除以 palm_size(手腕到中指根的 3D 距離),消除遠近差異。
    """
    points = np.array(landmarks)          # shape: (21, 3)
    wrist = points[0]                     # shape: (3,)

    palm_size = euclidean_distance_3d(landmarks[0], landmarks[9])
    if palm_size < 1e-6:                  # 防呆:避免除以接近 0(手部偵測異常時)
        palm_size = 1e-6

    translated = points - wrist           # broadcasting: (21,3) - (3,) → (21,3)
    scaled = translated / palm_size       # broadcasting: (21,3) / scalar

    return [tuple(point) for point in scaled]