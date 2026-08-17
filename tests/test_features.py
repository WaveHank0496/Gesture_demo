import pandas as pd
from src.gesture_demo.features import normalize

df = pd.read_csv("data/raw/gestures.csv", header=None)
row = df.iloc[0]
label = row[0]
landmarks = [(row[1+i*3], row[2+i*3], row[3+i*3]) for i in range(21)]

result = normalize(landmarks)

print("label:", label)
print("normalized[0] (應該接近 0,0,0):", result[0])
print("normalized[9] 到原點距離 (應該接近 1.0):",
      (result[9][0]**2 + result[9][1]**2 + result[9][2]**2) ** 0.5)