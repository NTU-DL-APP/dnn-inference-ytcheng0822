import numpy as np
import json
import struct
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# === 讀取 .ubyte 檔案 ===
def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# === 路徑 ===
images_path = "data/t10k-images-idx3-ubyte"
labels_path = "data/t10k-labels-idx1-ubyte"

# === 載入資料並正規化 ===
x = load_images(images_path).astype(np.float32) / 255.0
y = load_labels(labels_path)

# Optional: 分出一部分作為驗證集
val_split = 0.1
val_size = int(len(x) * val_split)

x_train, x_val = x[:-val_size], x[-val_size:]
y_train, y_val = y[:-val_size], y[-val_size:]

# 儲存 x_test.npy（讓 nn_predict.py 使用）
np.save("data/x_test.npy", x_val)

# === 建立模型 ===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# === 編譯與訓練 ===
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val))

# === 儲存模型 ===
model.save("fashion_mnist.h5")

# === 輸出 JSON 架構 ===
model_json = []
for i, layer in enumerate(model.layers):
    if layer.get_weights():
        model_json.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "config": layer.get_config(),
            "weights": [f"W{i}", f"b{i}"]
        })
    else:
        model_json.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "config": layer.get_config(),
            "weights": []
        })

os.makedirs("model", exist_ok=True)
with open("model/fashion_mnist.json", "w") as f:
    json.dump(model_json, f, indent=2)

# === 儲存權重為 .npz ===
weights = {}
for i, layer in enumerate(model.layers):
    w = layer.get_weights()
    if w:
        weights[f"W{i}"] = w[0]
        weights[f"b{i}"] = w[1]

np.savez("model/fashion_mnist.npz", **weights)