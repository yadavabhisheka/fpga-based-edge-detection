import cv2
import os

folder = 'software/test_images/'
TARGET = (640, 480)

for fname in os.listdir(folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, TARGET)
            cv2.imwrite(path, resized)
            print(f"{fname}: {orig_w}x{orig_h} → 640x480")

print("Done. All images resized.")