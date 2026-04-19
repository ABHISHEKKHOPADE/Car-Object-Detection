from src.model import build_model
import matplotlib.pyplot as plt
import os

def train_model(X, y_class, bbox, optimizer, save_path, history_path):
    model = build_model()

    model.compile(
        optimizer=optimizer,
        loss={
            "class_output": "binary_crossentropy",
            "bbox_output": "huber"
        },
        loss_weights={
            "class_output": 2.0,
            "bbox_output": 1.0
        },
        metrics={
            "class_output": "accuracy",
            "bbox_output": "mae"
        }
    )

    history = model.fit(
        X,
        {
            "class_output": y_class,
            "bbox_output": bbox
        },
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        shuffle=True
    )

    model.save(save_path)

    # Save accuracy history
    acc = history.history['class_output_accuracy']
    val_acc = history.history['val_class_output_accuracy']

    os.makedirs("results", exist_ok=True)

    with open(history_path, "w") as f:
        for a, v in zip(acc, val_acc):
            f.write(f"{a},{v}\n")

    return acc, val_acc