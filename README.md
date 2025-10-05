<img width="1918" height="1082" alt="image" src="https://github.com/user-attachments/assets/3c1bd2d0-1e7e-43b9-81ec-77aa83faf9db" />

This Hand Gesture Recognition model combines Convulation Neural Networks with Mediapipe hand landmarks.
We have 9 gestures at present - open palm, ok, left,right,up,down,index,fist, and one more...

## Project Structure
Hand Gesture/
│
├── .venv/                     # Virtual environment
│
├── dataset/                   # Dataset folder (contains gesture images)
│
├── .gitattributes
├── .gitignore
│
├── dataset_creation.ipynb     # Notebook for dataset preparation
├── model.ipynb                # Model training and evaluation
├── livecontrol.py             # Real-time gesture detection and control script
│
├── hand-gesture-model.keras   # Trained CNN + Landmark fusion model
├── hand_gestures.csv          # Gesture labels / metadata
├── le.pkl                     # Label encoder for gesture classes
├── scaler.pkl                 # Feature scaler for numerical normalization
│
└── README.md                  # Project documentation (this file)

## 🧠 Model Overview

The model uses a **hybrid deep learning approach**:
- **CNN** extracts **visual features** from raw gesture images.
- **MediaPipe** extracts **21-hand landmark coordinates** per frame.
- These two feature vectors are **combined** and passed through dense layers for classification.

### 🧩 CNN + ANN Architecture
```python
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
])
y_out = Dense(128, activation='relu')(combined_input)
y_out = Dropout(0.3)(y_out)
y_out = Dense(64, activation='relu')(y_out)
y_out = Dense(num_classes, activation='softmax')(y_out) 
```
## Compilation & Training
Optimizer: Adam(learning_rate=0.001)
Batch size: 1
Epochs: 17
Loss function: Categorical Crossentropy
Metrics: Accuracy

## workflow
Dataset Creation:
Run dataset_creation.ipynb to capture and label gesture images using your webcam and MediaPipe.
Model Training:
Use model.ipynb to train the CNN and fusion network on the prepared dataset.
Real-Time Detection:
Run livecontrol.py to use your webcam for live gesture prediction and real-time application control.

