import numpy as np
import cv2
import os
import tensorflow as tf


def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../model")
    return tf.keras.models.load_model(model_path)


def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, 0)
    frame = np.divide(frame, 255)
    frame = np.subtract(frame, 0.5)
    frame = np.multiply(frame, 2.0)
    return frame


def main(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        frame = preprocess_frame(frame)
        pred = model.predict(frame)
        print(pred)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    model = load_model()
    main(model)
