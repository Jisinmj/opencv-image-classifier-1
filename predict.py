import sys, joblib, cv2, numpy as np

IMG_SIZE = (64, 64)
MODEL_PATH = "image_classifier.pkl"

def predict_image(path):
    model = joblib.load(MODEL_PATH)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: image not found"); return
    img = cv2.resize(img, IMG_SIZE).flatten().astype("float32")/255.0
    pred = model.predict([img])[0]
    print("Dog 🐶" if pred == 1 else "Cat 🐱")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
    else:
        predict_image(sys.argv[1])
