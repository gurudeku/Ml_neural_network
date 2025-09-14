import random
from typing import List, Sequence, Tuple, Optional
import numpy as np
import pickle

import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime

def write_params_to_txt(net, path="weights_biases.txt", *, mode="w") -> None:
    """Write all weights and biases to a human-readable text file."""
    with open(path, mode, encoding="utf-8") as f:
        f.write(f"=== Snapshot: {datetime.now().isoformat()} ===\n")
        f.write(f"Layers: {net.sizes}\n")
        for i, (w, b) in enumerate(zip(net.weights, net.biases)):
            f.write(f"\nLayer {i}->{i+1}\n")
            f.write(f"W{i} shape: {w.shape}\n")
            f.write(np.array2string(w, precision=4, suppress_small=True))
            f.write("\n")
            f.write(f"b{i} shape: {b.shape}\n")
            f.write(np.array2string(b.ravel(), precision=4, suppress_small=True))
            f.write("\n")

Array = np.ndarray
TrainingPair = Tuple[Array, Array]   # (x, y) with shapes (n_in,1), (n_out,1)
TestPair = Tuple[Array, int]         # (x, class_index)

def run_gui(model_path="mnist.pkl"):
    # Load your trained model
    net = Network.load(model_path)
    write_params_to_txt(net, "weights_biases_on_gui_start.txt", mode="w")

    SCALE = 12            # draw at 28x larger resolution (28*12 = 336 px)
    CANVAS = 28 * SCALE
    BRUSH = 18           # brush thickness (tweak if strokes are too thin)

    # A PIL image we draw on (black background, white strokes)
    img = Image.new("L", (CANVAS, CANVAS), color=0)  # 'L' = grayscale
    draw = ImageDraw.Draw(img)

    root = tk.Tk()
    root.title("MNIST Draw & Predict (your NumPy net)")

    pred_var = tk.StringVar(value="Draw a digit, then click Predict")

    c = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black", highlightthickness=1, highlightbackground="#333")
    c.grid(row=0, column=0, columnspan=3, padx=8, pady=8)
    tk.Label(root, textvariable=pred_var, font=("Segoe UI", 12)).grid(row=1, column=0, columnspan=3, pady=(0,8))

    last = {"x": None, "y": None}

    def reset_last(_=None):
        last["x"], last["y"] = None, None

    def draw_motion(event):
        x, y = event.x, event.y
        if last["x"] is None:
            last["x"], last["y"] = x, y
        # draw on canvas (visual)
        c.create_line(last["x"], last["y"], x, y, fill="white", width=BRUSH, capstyle=tk.ROUND, smooth=True)
        # draw on PIL image (data)
        draw.line((last["x"], last["y"], x, y), fill=255, width=BRUSH)
        last["x"], last["y"] = x, y

    def clear():
        c.delete("all")
        draw.rectangle((0, 0, CANVAS, CANVAS), fill=0)
        pred_var.set("Cleared. Draw a digit, then click Predict.")

    def predict():
        # downscale to 28x28 like MNIST
        img28 = img.resize((28, 28))
        arr = np.array(img28, dtype=np.float32) / 255.0   # 0..1
        x = arr.reshape(784, 1)                           # column vector
        out = net.feedforward(x)                          # (10,1)
        pred = int(np.argmax(out))
        conf = float(out[pred, 0])
        pred_var.set(f"Prediction: {pred}   (output={conf:.3f})")

    c.bind("<B1-Motion>", draw_motion)
    c.bind("<ButtonRelease-1>", reset_last)

    tk.Button(root, text="Predict", command=predict).grid(row=2, column=0, padx=8, pady=8, sticky="ew")
    tk.Button(root, text="Clear", command=clear).grid(row=2, column=1, padx=8, pady=8, sticky="ew")
    tk.Button(root, text="Close", command=root.destroy).grid(row=2, column=2, padx=8, pady=8, sticky="ew")

    root.mainloop()
def sigmoid(z: Array) -> Array:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: Array) -> Array:
    """Derivative of the sigmoid."""
    s = sigmoid(z)
    return s * (1 - s)


class Network:
    def __init__(self, sizes: Sequence[int]) -> None:
        """
        sizes: layer sizes, e.g. [2, 3, 1] for a 2-3-1 network.
        Weights/biases ~ N(0,1). No biases for input layer.
        """
        self.num_layers: int = len(sizes)
        self.sizes: List[int] = list(sizes)
        self.biases: List[Array] = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights: List[Array] = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a: Array) -> Array:
        """Return the network output for input column vector a."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(
        self,
        training_data: List[TrainingPair],
        epochs: int,
        mini_batch_size: int,
        eta: float,
        test_data: Optional[List[TestPair]] = None,
        *,
        shuffle: bool = True,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Mini-batch stochastic gradient descent.
        training_data: list of (x, y) where x,y are column vectors.
        test_data: list of (x, label_index) for evaluation.
        """
        if rng is None:
            rng = random

        n = len(training_data)
        n_test = len(test_data) if test_data is not None else None

        for j in range(epochs):
            if shuffle:
                rng.shuffle(training_data)

            # Split into mini-batches
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Update per mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Progress
            if test_data is not None:
                acc = self.evaluate(test_data)
                print(f"Epoch {j}: {acc} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch: List[TrainingPair], eta: float) -> None:
        """Gradient step using backprop over one mini-batch."""
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        m = len(mini_batch)
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x: Array, y: Array) -> Tuple[List[Array], List[Array]]:
        """
        Return (nabla_b, nabla_w) for a single (x,y):
          nabla_b[l] = ∂C/∂b^l, nabla_w[l] = ∂C/∂W^l
        """
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        # Forward pass
        activation = x
        activations = [x]  # list of layer activations
        zs: List[Array] = []  # list of z vectors
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass (quadratic cost)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # l = 2 means the second-last layer, etc.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def evaluate(self, test_data: List[TestPair]) -> int:
        """
        Return count of correct predictions on test_data.
        test_data labels are integer class indices.
        """
        test_results = [(int(np.argmax(self.feedforward(x))), y) for (x, y) in test_data]
        return sum(int(pred == y) for pred, y in test_results)

    @staticmethod
    def cost_derivative(output_activations: Array, y: Array) -> Array:
        """∂C/∂a for quadratic cost."""
        return output_activations - y
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"sizes": self.sizes, "weights": self.weights, "biases": self.biases}, f)

    @classmethod
    def load(cls, path: str) -> "Network":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        net = cls(obj["sizes"])
        net.weights = obj["weights"]
        net.biases = obj["biases"]
        return net
def one_hot(j: int, n: int = 10) -> np.ndarray:
    e = np.zeros((n,1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    import sys

    # Usage:
    #   python net.py train    -> trains on MNIST and saves mnist.pkl
    #   python net.py gui      -> opens drawing window using mnist.pkl
    mode = sys.argv[1] if len(sys.argv) > 1 else "gui"

    if mode == "train":
        # NOTE: TensorFlow/keras needs Python 3.12/3.11 (not 3.13)
        from tensorflow.keras.datasets import mnist

        def one_hot(j: int, n: int = 10) -> np.ndarray:
            e = np.zeros((n,1)); e[j] = 1.0; return e

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 784, 1).astype("float32") / 255.0
        x_test  = x_test.reshape(-1, 784, 1).astype("float32") / 255.0

        training_data = [(x, one_hot(int(y))) for x, y in zip(x_train, y_train)]
        test_data = [(x, int(y)) for x, y in zip(x_test, y_test)]

        net = Network([784, 30, 10])
        net.SGD(training_data, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data)
        net.save("mnist.pkl")
        print("Saved model to mnist.pkl")

    elif mode == "gui":
        run_gui("mnist.pkl")   # requires a saved model
    else:
        print("Unknown mode. Use: python net.py train  OR  python net.py gui")



