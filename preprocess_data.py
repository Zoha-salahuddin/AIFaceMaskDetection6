# scripts/preprocess_data.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset"
IMG_SIZE = 224

data = []
labels = []

def load_images(folder, label):
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
        labels.append(label)

load_images(os.path.join(DATASET_PATH, "with_mask"), 1)
load_images(os.path.join(DATASET_PATH, "without_mask"), 0)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

os.makedirs("model", exist_ok=True)
np.save("model/X_train.npy", X_train)
np.save("model/X_test.npy", X_test)
np.save("model/y_train.npy", y_train)
np.save("model/y_test.npy", y_test)

print("✅ Preprocessing completed")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
