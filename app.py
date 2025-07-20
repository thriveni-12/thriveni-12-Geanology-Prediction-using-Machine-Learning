from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained model
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

model = create_model()

# Function for face detection and cropping
def detect_and_crop_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return img  # Return original if no face is detected

    # Crop the first detected face
    (x, y, w, h) = faces[0]
    cropped_face = img[y:y + h, x:x + w]
    cropped_face_resized = cv2.resize(cropped_face, (224, 224))
    return cropped_face_resized

# Function to preprocess the image, crop face, and extract features
def extract_features(img_path):
    cropped_face = detect_and_crop_face(img_path)
    cropped_face_array = image.img_to_array(cropped_face)
    cropped_face_array = np.expand_dims(cropped_face_array, axis=0)
    cropped_face_array = preprocess_input(cropped_face_array)  # Standardize as per VGG16 requirements
    
    features = model.predict(cropped_face_array)
    return features.flatten()

# Matching features based on similarity scores for different traits
# Matching features based on similarity scores for each trait
def find_matching_characteristics(child_features, **parent_features):
    # List of traits and their descriptions
    traits = {
        "Eye Shape": "Shape similarity",
        "Eye Size": "Size match",
        "Eye Position": "Position alignment",
        "Eye Angle": "Angle correlation",
        "Eye Distance": "Distance match",
        "Eyebrow Shape": "Shape definition",
        "Eyebrow Thickness": "Thickness consistency",
        "Eyebrow Distance": "Distance match",
        "Eyebrow Arch": "Arch shape",
        "Skin Tone": "Pigmentation match",
        "Skin Texture": "Texture alignment",
        "Freckles": "Freckle pattern match",
        "Mole Pattern": "Mole pattern similarity",
        "Birthmarks": "Birthmark resemblance",
        "Nose Shape": "Shape resemblance",
        "Nose Size": "Size similarity",
        "Nose Width": "Width consistency",
        "Nose Bridge": "Bridge alignment",
        "Nose Tip": "Tip shape",
        "Lip Fullness": "Fullness consistency",
        "Lip Shape": "Shape symmetry",
        "Lip Symmetry": "Symmetry alignment",
        "Lip Curve": "Curve definition",
        "Lip Color": "Color match",
        "Cheekbone Height": "Height alignment",
        "Cheekbone Prominence": "Prominence definition",
        "Cheekbone Width": "Width match",
        "Chin Shape": "Shape similarity",
        "Chin Size": "Size consistency",
        "Chin Protrusion": "Protrusion resemblance",
        "Jawline Shape": "Shape definition",
        "Jawline Width": "Width alignment",
        "Jawline Angle": "Angle match",
        "Forehead Shape": "Shape symmetry",
        "Forehead Height": "Height similarity",
        "Forehead Width": "Width alignment",
        "Ear Shape": "Shape consistency",
        "Ear Size": "Size resemblance",
        "Ear Position": "Position alignment",
        "Ear Angle": "Angle match",
        "Face Shape": "Overall shape symmetry",
        "Face Symmetry": "Symmetry definition",
        "Face Width": "Width similarity",
        "Face Length": "Length alignment",
        "Hairline Shape": "Shape symmetry",
        "Hairline Height": "Height match",
        "Hairline Symmetry": "Symmetry alignment",
        "Neck Length": "Length resemblance",
        "Neck Width": "Width alignment",
        "Neck Shape": "Shape consistency",
        "Smile Curve": "Curve resemblance",
        "Smile Width": "Width alignment",
        "Smile Symmetry": "Symmetry definition",
        "Dimple Position": "Position alignment",
        "Dimple Depth": "Depth match",
        "Dimple Symmetry": "Symmetry resemblance",
        "Eye Bags": "Bag prominence",
        "Eye Wrinkles": "Wrinkle alignment",
        "Nose Wrinkles": "Wrinkle definition",
    }

    # Thresholds for each trait
    trait_thresholds = {trait: 0.4 for trait in traits.keys()}
    trait_thresholds.update({"Eye Wrinkles": 0.5, "Nose Wrinkles": 0.5, "Freckles": 0.5, "Birthmarks": 0.5})

    matching_characteristics = {relation: [] for relation in parent_features.keys()}

    # Evaluate similarity for each parent
    for relation, features in parent_features.items():
        if features is not None:
            for trait, description in traits.items():
                # Compute similarity (Euclidean distance in this example)
                similarity = np.linalg.norm(child_features - features)

                # Evaluate trait match
                if similarity < trait_thresholds[trait]:
                    matching_characteristics[relation].append(f"{trait}: Matched ({description})")
                else:
                    matching_characteristics[relation].append(f"{trait}: Partially matched (similarity score: {similarity:.2f})")

    return matching_characteristics

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_images = {}
    features = {}
    relations = ["grandmother", "grandfather", "mother", "father", "child"]

    for relation in relations:
        image_file = request.files.get(relation)
        if image_file and image_file.filename != '':
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            uploaded_images[relation] = filename
            features[relation] = extract_features(filepath)

    if "child" not in features or ("father" not in features and "mother" not in features):
        return "Please upload at least the child's image and one parent (father or mother) image."

    # Extract matching characteristics for specific traits
    child_features = features["child"]
    matching_characteristics = find_matching_characteristics(
        child_features,
        grandmother=features.get("grandmother"),
        grandfather=features.get("grandfather"),
        mother=features.get("mother"),
        father=features.get("father")
    )

    # Calculate similarity scores for uploaded images only
    similarities = {relation: np.linalg.norm(features[relation] - child_features) for relation in features if relation != "child"}

    return render_template(
        'results.html',
        uploaded_images=uploaded_images,
        similarities=similarities,
        matching_characteristics=matching_characteristics
    )

if __name__ == '__main__':
    app.run(debug=True)
