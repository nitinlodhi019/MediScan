# Medical AI App (FastAPI + Keras)

This is a minimal FastAPI web app that loads your trained Keras models (.keras or .h5) and serves
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

2) Copy your trained models into `app/models/` and keep (or update) the names in `app/config.py`:
   - `xray_model.h5` or `xray_model.keras`
   - `brain_model.h5` or `brain_model.keras`
   (Add `skin_model.h5` later if you train it.)

3) Run the app locally:
   ```bash
   uvicorn app.main:app --reload
   ```
   Open http://127.0.0.1:8000 in your browser.

4) Choose a model (X-ray / Brain), upload an image, and see the formatted report.

## Notes
- Uses Keras 3 API (compatible with `.keras` and `.h5`).
- If you face version issues, ensure your local `keras` + `tensorflow` are similar to what you used on Kaggle.
- Default image size is 224x224. Adjust in `app/services/preprocessing.py` if different.
