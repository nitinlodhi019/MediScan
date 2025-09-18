# Medical AI App (Flask + Keras)

This is a minimal Flask web app that loads your trained Keras models (.h5) and serves
a simple frontend for image upload and prediction. It returns a formatted clinical-style report
with prediction, confidence, findings, and recommendations.

## 📂 Project Structure

```bash
MediScan/
├── app.py                  
├── disease_info.json        
├── preprocess.py           
├── requirements.txt        
├── README.md             
├── .gitignore         
├── models/                 
│   ├── brain_tumor_xception_model.h5
│   ├── Chest_XRay.h5
│   ├── Chest_XRay_model.h5
│   ├── fracture_classification_model.h5
│   ├── gatekeeper_model.h5
│   └── resnet50_binary_softmax.h5
├── templates/          
│   └── index.html
├── venv/                    # Virtual environment
│   └── ...
└── Lib/                     # Auto-generated Python libraries
```

## Some Images

<img width="1819" height="915" alt="Screenshot 2025-09-18 202044" src="https://github.com/user-attachments/assets/f5d2c770-d5cd-4d1b-9aa1-7e428f5c7288" />

<img width="1820" height="927" alt="Screenshot 2025-09-18 202056" src="https://github.com/user-attachments/assets/5100c1c1-f3ee-4ab1-9c84-9a91fe7c7658" />


## Run Locally
- Clone the project

- Create a virtual environment
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```

- Install Dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
- Go to saved_models directory using
   ```bash
   cd models
   ```
   - Download the PreTrained models and model weights from https://drive.google.com/drive/folders/1_7OAB1r3PCBxUS0UpyxkqrKQbVgln0IH?usp=drive_link
   - Paste the downloaded weights in the corresponding directories.

- Go back to the parent directory using
   ```bash
   cd ../
   ```

- Run the app locally:
   ```bash
   python app.py
   ```
   Open http://127.0.0.1:8000 in your browser.

## Notes
- Uses Keras 3 API (compatible with `.keras` and `.h5`).
- If you face version issues, ensure your local `keras` + `tensorflow` are similar to what you used on Kaggle.
- Default image size is 224x224. Adjust in `app/services/preprocessing.py` if different.
