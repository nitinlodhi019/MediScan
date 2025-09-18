# Medical AI App (Flask + Keras)

This is a minimal Flask web app that loads your trained Keras models (.h5) and serves
a simple frontend for image upload and prediction. It returns a formatted clinical-style report
with prediction, confidence, findings, and recommendations.

## Quickstart

1) Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2) Copy your trained models into `../models/`:
   - `brain_tumor_xception_model.h5`
   - `Chest_XRay_model.h5`
   - `fracture_classification_model.h5`
   - `gatekeeper_model.h5`
   - `resnet50_binary_softmax.h5

3) Run the app locally:
   ```bash
   python app.py
   ```
   Open http://127.0.0.1:8000 in your browser.

## Notes
- Uses Keras 3 API (compatible with `.keras` and `.h5`).
- If you face version issues, ensure your local `keras` + `tensorflow` are similar to what you used on Kaggle.
- Default image size is 224x224. Adjust in `app/services/preprocessing.py` if different.
