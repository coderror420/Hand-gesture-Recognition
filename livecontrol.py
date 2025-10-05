import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf

# loading model,scaler,le
model = tf.keras.models.load_model("hand-gesture-model.keras")
scaler = joblib.load("scaler.pkl")
le = joblib.load("le.pkl")

# mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

IMG_SIZE = (256, 256)

# preprocessing
def preprocess_image_rgb(frame_rgb, hand_landmarks):
    """Crop hand region from RGB frame using landmarks and resize"""
    h, w, _ = frame_rgb.shape
    x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

    # Bounding box with padding
    x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, w)
    y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, h)

    hand_img = frame_rgb[y_min:y_max, x_min:x_max]
    if hand_img.size == 0:  # fallback
        hand_img = frame_rgb

    hand_img = cv2.resize(hand_img, IMG_SIZE)
    hand_img = hand_img / 255.0
    return np.expand_dims(hand_img, axis=0)  # (1, 256, 256, 3)

def preprocess_landmarks(hand_landmarks, width, height):
    """Extract and scale hand landmarks"""
    lm = []
    for lmks in hand_landmarks.landmark:
        lm.append(lmks.x )
        lm.append(lmks.y )
        lm.append(lmks.z )
    lm = np.array(lm).reshape(1, -1)
    return scaler.transform(lm)

#predict gestures
def predict_gesture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB for prediction (match training set)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                # Preprocess
                img_inp = preprocess_image_rgb(frame_rgb, hand_landmarks)
                lm_inp = preprocess_landmarks(hand_landmarks, w, h)

                # Predict
                pred = model.predict([img_inp, lm_inp])
                pred_idx = np.argmax(pred, axis=1)
                pred_label = le.inverse_transform(pred_idx)[0]

                # Show prediction
                cv2.putText(frame, pred_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Hand", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- Run -------------------
if __name__ == "__main__":
    predict_gesture()
