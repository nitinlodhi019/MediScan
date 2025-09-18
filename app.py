from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import VarianceScaling, Zeros, Ones, GlorotUniform, GlorotNormal
import numpy as np
import json
from PIL import Image

from preprocess import (
    preprocess_chest_xray,
    preprocess_mri_brain_tumor,
    preprocess_skin_cancer,
    preprocess_bone_fracture
)

CONFIDENCE_THRESHOLD = 75.0

app = Flask(__name__)

class SafeVarianceScaling(VarianceScaling):
    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None, **kwargs):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)

class SafeZeros(Zeros):
    def __init__(self, **kwargs):
        super().__init__()

class SafeOnes(Ones):
    def __init__(self, **kwargs):
        super().__init__()

class SafeGlorotUniform(GlorotUniform):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed)

class SafeGlorotNormal(GlorotNormal):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed)

# --- Custom Objects Map ---
custom_objects = {
    'VarianceScaling': SafeVarianceScaling,
    'Zeros': SafeZeros,
    'Ones': SafeOnes,
    'GlorotUniform': SafeGlorotUniform,
    'GlorotNormal': SafeGlorotNormal,
    'BatchNormalizationV1': BatchNormalization
}

# 1. Gatekeeper Model
gatekeeper_model = load_model('models/gatekeeper_model.h5')
gatekeeper_labels = ['brain_tumor', 'chest_xray', 'skin_cancer', 'bone_fracture', 'other']

# 2. Chest X-ray Model
model_chest_xray = load_model('models/Chest_XRay_model.h5')

# 3. Brain Tumor Model
model_brain_tumor = load_model("models/brain_tumor_xception_model.h5", compile=False)

# 4. Skin Cancer Model
model_skin_cancer = load_model(
    'models/resnet50_binary_softmax.h5',
    compile=False,
    custom_objects=custom_objects
)

# 5. Bone Fracture Model
model_bone_fracture = load_model('models/fracture_classification_model.h5', compile=False)

MODELS_CONFIG = {
    'chest_xray': {
        'model': model_chest_xray,
        'labels': ['PNEUMONIA', 'NORMAL'],
        'preprocess_func': preprocess_chest_xray,
        'type': 'binary'
    },
    'brain_tumor': {
        'model': model_brain_tumor,
        'labels': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'preprocess_func': preprocess_mri_brain_tumor,
        'type': 'multiclass'
    },
    'skin_cancer': {
        'model': model_skin_cancer,
        'labels': ['Benign', 'Malignant'],
        'preprocess_func': preprocess_skin_cancer,
        'type': 'binary'
    },
    'bone_fracture': {
        'model': model_bone_fracture,
        'labels': ['fractured', 'not fractured'],
        'preprocess_func': preprocess_bone_fracture,
        'type': 'binary'
    }
}

#Load disease info
with open('disease_info.json', 'r') as f:
    disease_info = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    file_path = 'uploaded_image.jpg'
    file.save(file_path)

    try:
        img = Image.open(file_path).convert('L').resize((150, 150))
        img_array = (np.array(img) / 255.0).reshape(1, 150, 150, 1)

        gatekeeper_preds = gatekeeper_model.predict(img_array)[0]
        image_class_index = np.argmax(gatekeeper_preds)
        image_type = gatekeeper_labels[image_class_index]

        print("🔎 Gatekeeper raw predictions:", gatekeeper_preds)
        print("👉 Predicted type:", image_type)

        gatekeeper_debug = {
            "raw_predictions": {label: float(score) for label, score in zip(gatekeeper_labels, gatekeeper_preds)},
            "chosen_class": image_type
        }

    except Exception as e:
        return jsonify({'error': f"Gatekeeper analysis failed: {str(e)}"}), 500

    if image_type == 'other':
        return jsonify({'rejection': True, 'message': 'Image Type Not Supported'}), 200

    # Specialist Prediction
    try:
        config = MODELS_CONFIG[image_type]
        preprocessed_image = config['preprocess_func'](file_path)
        predictions = config['model'].predict(preprocessed_image)[0]

        if config['type'] == 'binary':
            pred_val = predictions[0]
            pred_idx = 1 if pred_val > 0.5 else 0
            confidence = pred_val if pred_idx == 1 else 1 - pred_val
        else:
            pred_idx = np.argmax(predictions)
            confidence = predictions[pred_idx]

        predicted_disease = config['labels'][pred_idx]
        confidence_score = float(confidence) * 100

        if confidence_score < CONFIDENCE_THRESHOLD:
            return jsonify({
                'warning': 'Low Confidence Prediction',
                'predicted_disease': predicted_disease,
                'confidence_score': confidence_score,
                'key_findings': ["Model confidence is below the threshold for a reliable diagnosis."],
                'recommendation': ["Please use a clearer or more standard image and try again."]
            })

        info = disease_info.get(predicted_disease.lower(), {})
        return jsonify({
            'predicted_disease': predicted_disease,
            'confidence_score': confidence_score,
            'key_findings': info.get('key_findings', []),
            'recommendation': info.get('recommendation', [])
        })
    except Exception as e:
        return jsonify({'error': f"Diagnosis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
