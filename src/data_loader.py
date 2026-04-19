import os
import cv2
import pandas as pd
import numpy as np
import random
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 224

def load_data(image_dir, csv_file, negative_dir):
    df = pd.read_csv(csv_file)

    images, bboxes, labels = [], [], []

    #  POSITIVE SAMPLES
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        xmin = row['xmin'] / w
        ymin = row['ymin'] / h
        xmax = row['xmax'] / w
        ymax = row['ymax'] / h

        bbox = [xmin, ymin, xmax, ymax]

        # Augmentation
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            xmin, ymin, xmax, ymax = bbox
            bbox = [1 - xmax, ymin, 1 - xmin, ymax]

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img)

        images.append(img)
        bboxes.append(bbox)
        labels.append(1)

    # BALANCED NEGATIVE SAMPLES
    neg_files = os.listdir(negative_dir)
    neg_files = neg_files[:len(labels)]  # balance

    for file in neg_files:
        img_path = os.path.join(negative_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img)

        images.append(img)
        bboxes.append([0,0,0,0])
        labels.append(0)

    X = np.array(images, dtype="float32")
    y_class = np.array(labels, dtype="float32").reshape(-1,1)
    y_bbox = np.array(bboxes, dtype="float32")

    return X, y_class, y_bbox