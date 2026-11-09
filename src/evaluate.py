import json, numpy as np, matplotlib
matplotlib.use("Agg")
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
d = np.load("data/processed/data.npz")
m = keras.models.load_model("models/model.h5")
loss, acc = m.evaluate(d["x_test"], d["y_test"])
json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"))
pred = m.predict(d["x_test"]).argmax(1)
ConfusionMatrixDisplay.from_predictions(d["y_test"], pred).figure_.savefig("confusion_matrix.png")