import joblib
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os, glob

IMG_SIZE = (64, 64)

def load_images(folder, label):
    X, y = [], []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for fp in glob.glob(os.path.join(folder, ext)):
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            img = cv2.resize(img, IMG_SIZE).flatten().astype("float32")/255.0
            X.append(img); y.append(label)
    return np.array(X), np.array(y)

X_cats, y_cats = load_images("data/cats", 0)
X_dogs, y_dogs = load_images("data/dogs", 1)

X = np.vstack([X_cats, X_dogs])
y = np.concatenate([y_cats, y_dogs])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearSVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred)*100:.2f}%")

joblib.dump(model, "image_classifier.pkl")
print("✅ Model saved -> image_classifier.pkl")
