import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Load trained CNN model
model = load_model('smartbin_cnn_model.keras')

# ✅ Set image size (match training size)
IMG_SIZE = (128, 128)

# ✅ Labels
class_labels = ['Non-Recyclable', 'Recyclable']

# ✅ Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found.")
    exit()

print("📸 Press 'c' to capture image for prediction")
print("❌ Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    # Show live feed
    cv2.imshow("♻️ SmartBin - Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # ✅ Preprocess image
        image = cv2.resize(frame, IMG_SIZE)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # ✅ Predict
        prediction = model.predict(image)[0][0]
        label = class_labels[1] if prediction >= 0.5 else class_labels[0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        print(f"🧠 Prediction: {label} ({confidence * 100:.2f}%)")

        # ✅ Display result
        result_frame = frame.copy()
        cv2.putText(result_frame, f"{label} ({confidence * 100:.2f}%)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0) if label == "Recyclable" else (0, 0, 255), 3)
        cv2.imshow("♻️ SmartBin - Result", result_frame)
        cv2.waitKey(3000)  # Show result for 3 seconds

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
