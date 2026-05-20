from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import mne
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("sleep_model.pkl")
scaler = joblib.load("scaler.pkl")

sleep_labels = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "Deep",
    4: "REM"
}

@app.get("/")
def home():
    return {
        "message": "Sleep Detection API Running Successfully"
    }

@app.post("/predict")
async def predict_sleep(file: UploadFile = File(...)):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:

            content = await file.read()

            tmp.write(content)

            temp_path = tmp.name

        raw = mne.io.read_raw_edf(
            temp_path,
            preload=True,
            verbose=False
        )

        signal = raw.get_data()[0]

        signal = signal[:3000]

        freqs, psd = welch(
            signal,
            fs=raw.info['sfreq']
        )

        delta = np.mean(
            psd[(freqs >= 0.5) & (freqs < 4)]
        )

        theta = np.mean(
            psd[(freqs >= 4) & (freqs < 8)]
        )

        alpha = np.mean(
            psd[(freqs >= 8) & (freqs < 13)]
        )

        beta = np.mean(
            psd[(freqs >= 13) & (freqs < 30)]
        )

        features = [[
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            skew(signal),
            kurtosis(signal),
            delta,
            theta,
            alpha,
            beta
        ]]

        X_scaled = scaler.transform(features)

        prediction = model.predict(X_scaled)[0]

        result = sleep_labels.get(
            int(prediction),
            "Unknown"
        )

        return {
            "predicted_stage": result
        }

    except Exception as e:

        return {
            "error": str(e)
        }

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)