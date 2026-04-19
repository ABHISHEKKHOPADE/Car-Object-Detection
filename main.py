from src.data_loader import load_data
from src.train import train_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os

os.makedirs("models", exist_ok=True)

X, y_class, bbox = load_data(
    "data/training_images",
    "data/train_solution_bounding_boxes.csv",
    "data/negatives"
)

# Train models
adam_acc, adam_val = train_model(
    X, y_class, bbox,
    Adam(0.001),
    "models/adam.h5",
    "results/adam.txt"
)

sgd_acc, sgd_val = train_model(
    X, y_class, bbox,
    SGD(0.01, momentum=0.9),
    "models/sgd.h5",
    "results/sgd.txt"
)

rms_acc, rms_val = train_model(
    X, y_class, bbox,
    RMSprop(0.001),
    "models/rmsprop.h5",
    "results/rmsprop.txt"
)