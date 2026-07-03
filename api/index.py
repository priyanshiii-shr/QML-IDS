from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="QML-IDS-IoT API", version="1.0")

# Enable CORS for frontend hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "random_forest.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "results", "models", "scaler.joblib")
FN_PATH = os.path.join(BASE_DIR, "results", "models", "feature_names.joblib")

# Lazy load models to minimize cold start times
model = None
scaler = None
feature_names = None

def load_assets():
    global model, scaler, feature_names
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FN_PATH)

class PredictRequest(BaseModel):
    duration: float = 0
    protocol_type: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: float = 0
    dst_bytes: float = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: float = 0
    srv_count: float = 0
    serror_rate: float = 0
    srv_serror_rate: float = 0
    rerror_rate: float = 0
    srv_rerror_rate: float = 0
    same_srv_rate: float = 0
    diff_srv_rate: float = 0
    srv_diff_host_rate: float = 0
    dst_host_count: float = 0
    dst_host_srv_count: float = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0
    dst_host_same_src_port_rate: float = 0
    dst_host_srv_diff_host_rate: float = 0
    dst_host_serror_rate: float = 0
    dst_host_srv_serror_rate: float = 0
    dst_host_rerror_rate: float = 0
    dst_host_srv_rerror_rate: float = 0

@app.get("/api/health")
def health_check():
    try:
        load_assets()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "features_count": len(feature_names) if feature_names else 0
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        load_assets()
        
        # Build raw dict structure initialized to 0
        input_dict = {col: 0 for col in feature_names}
        
        # Pop standard numeric features
        req_dict = req.dict()
        for k, v in req_dict.items():
            if k in ["protocol_type", "service", "flag"]:
                continue
            if k in input_dict:
                input_dict[k] = v
                
        # One-hot encode incoming strings
        proto_col = f"protocol_type_{req.protocol_type}"
        if proto_col in input_dict:
            input_dict[proto_col] = 1
            
        svc_col = f"service_{req.service}"
        if svc_col in input_dict:
            input_dict[svc_col] = 1
            
        flag_col = f"flag_{req.flag}"
        if flag_col in input_dict:
            input_dict[flag_col] = 1
            
        # Re-index to ensure perfect feature schema match
        df = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
        
        # Scale
        X_scaled = scaler.transform(df)
        
        # Classify
        pred = int(model.predict(X_scaled)[0])
        prob_atk = float(model.predict_proba(X_scaled)[0][1])
        
        return {
            "prediction": pred,
            "attack_probability": prob_atk,
            "normal_probability": 1.0 - prob_atk,
            "label": "Attack" if pred == 1 else "Normal",
            "status": "success"
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
