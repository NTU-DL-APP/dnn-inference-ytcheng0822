import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Softmax with numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward pass ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

# === Inference entry point ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)

# === Optional: Entry point for testing ===
if __name__ == "__main__":
    # Load model architecture
    with open("model/fashion_mnist.json", "r") as f:
        model_arch = json.load(f)

    # Load weights
    weights_npz = np.load("model/fashion_mnist.npz")
    weights = {key: weights_npz[key] for key in weights_npz.files}

    # Load and normalize test data (example path)
    x_test = np.load("data/x_test.npy")  # shape: (N, 28, 28)
    x_test = x_test.astype(np.float32) / 255.0

    # Inference
    logits = nn_inference(model_arch, weights, x_test)
    predictions = np.argmax(logits, axis=1)

    # Save or print predictions
    print("Predictions:", predictions)