import os, yaml, numpy as np
from sklearn.model_selection import train_test_split
p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion.npz")
<<<<<<< HEAD
x, xt = np.clip(d["x_train"] / 255.0, 0, 1), np.clip(d["x_test"] / 255.0, 0, 1)
=======
x, xt = d["x_train"].astype("float32") / 255.0, d["x_test"].astype("float32") / 255.0
>>>>>>> teammate-sim
xtr, xv, ytr, yv = train_test_split(x, d["y_train"], test_size=p["test_size"], random_state=p["seed"])
os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/data.npz", x_train=xtr, y_train=ytr, x_val=xv, y_val=yv, x_test=xt, y_test=d["y_test"])