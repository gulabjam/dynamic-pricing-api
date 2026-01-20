from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from train_model import load_model, predict_demand_with_model
import uvicorn
import os
from pathlib import Path

# ============ Model Loading at Startup ============
def ensure_model_exists():
    """Download model from HuggingFace if not present locally"""
    model_path = Path("demand_prediction_model.pkl")
    
    if not model_path.exists():
        print("⚠️ Model file not found locally. Attempting to download from HuggingFace...")
        
        # Change this to your HuggingFace repo after uploading
        HF_REPO_ID = os.getenv("HF_REPO_ID", "bhavin273/dynamic-parking-demand-model")
        
        try:
            from huggingface_hub import hf_hub_download
            
            print(f"📥 Downloading model from {HF_REPO_ID}...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="demand_prediction_model.pkl",
                local_dir=".",
                local_dir_use_symlinks=False
            )
            print("✅ Model downloaded from HuggingFace successfully!")
        except Exception as e:
            print(f"❌ Could not download model from HuggingFace: {e}")
            print("💡 Make sure:")
            print("   1. You've uploaded the model using: python huggingface_upload.py")
            print("   2. Set HF_REPO_ID environment variable to your repo")
            print("   3. Or place demand_prediction_model.pkl in the root directory")
            raise RuntimeError("Model file not available")
    else:
        print(f"✅ Model file found at {model_path}")

# Ensure model exists and load it once at startup
try:
    ensure_model_exists()
    MODEL_DATA = load_model()
    print("🚀 Model loaded and ready for predictions!")
except Exception as e:
    print(f"❌ Error during model initialization: {e}")
    MODEL_DATA = None
# ==================================================

app = FastAPI(title="Dynamic Parking Demand Predictor API")

class PredictRequest(BaseModel):
    h3_cell: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "h3_cell": "8c2a100d1a2bfff",
                "timestamp": "2026-01-19 14:00:00"
            }
        }

class PredictResponse(BaseModel):
    h3_cell: str
    timestamp: str
    demand_factor: float

class HealthResponse(BaseModel):
    status: str

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict demand factor for a given h3_cell and timestamp.
    
    Expected JSON payload:
    {
        "h3_cell": "string",
        "timestamp": "YYYY-MM-DD HH:MM:SS" or datetime string
    }
    
    Returns:
    {
        "h3_cell": "string",
        "timestamp": "string",
        "demand_factor": float
    }
    """
    try:
        # Check if model is loaded
        if MODEL_DATA is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        # Convert timestamp to datetime
        timestamp = pd.to_datetime(request.timestamp).round('H')
        
        # Predict demand using pre-loaded model
        demand_factor = predict_demand_with_model(request.h3_cell, timestamp, MODEL_DATA)
        
        if demand_factor is None:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for h3_cell: {request.h3_cell} at timestamp: {request.timestamp}"
            )
        
        return PredictResponse(
            h3_cell=request.h3_cell,
            timestamp=str(timestamp),
            demand_factor=demand_factor
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health', response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return HealthResponse(status="healthy")

@app.get('/')
def root():
    """Root endpoint with API information"""
    return {
        "message": "Dynamic Parking Demand Predictor API",
        "endpoints": {
            "POST /predict": "Predict demand factor",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }