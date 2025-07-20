from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2

# Create and load the VGG16 model
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

model = create_model()

# Detect and crop face
def detect_and_crop_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return img  # Return original if no face is detected

    (x, y, w, h) = faces[0]
    cropped_face = img[y:y + h, x:x + w]
    return cv2.resize(cropped_face, (224, 224))

# Extract features from the image
def extract_features(img_path):
    cropped_face = detect_and_crop_face(img_path)
    cropped_face_array = image.img_to_array(cropped_face)
    cropped_face_array = np.expand_dims(cropped_face_array, axis=0)
    cropped_face_array = preprocess_input(cropped_face_array)
    features = model.predict(cropped_face_array)
    return features.flatten()
