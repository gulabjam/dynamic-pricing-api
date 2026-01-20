import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pickle
from datetime import datetime
from db_client import get_db_connection, COLLECTION

# ================= CONFIG =================
MODEL_PATH = "demand_prediction_model.pkl"
RANDOM_STATE = 42
le = LabelEncoder()
# =========================================


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def export_data_from_mongo():
    print("🔹 Exporting data from MongoDB...")
    with get_db_connection() as db:
        collection = db[COLLECTION]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
        print(f"✅ Loaded {len(df)} rows from MongoDB")
        return df


# ------------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------------
def per_h3_time_split(data):
    train_raw = []
    val_raw = []
    val_true_list = []

    for h3_cell, group in data.groupby("h3_cell"):
        group = group.sort_values("timestamp")
        val_part = group.tail(24).copy()
        train_part = group.iloc[:-24].copy()

        val_true_list.append(val_part[['h3_cell', 'timestamp', 'demand']].copy())
        val_part["demand"] = np.nan

        train_raw.append(train_part)
        val_raw.append(val_part)

    return (
        pd.concat(train_raw).reset_index(drop=True),
        pd.concat(val_raw).reset_index(drop=True),
        pd.concat(val_true_list).reset_index(drop=True),
    )


# ------------------------------------------------------------------
# TRAINING FEATURE PIPELINE (HISTORICAL DATA ONLY)
# ------------------------------------------------------------------
def prepare_training_features(data):
    print("Preparing TRAINING features...")

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["h3_cell", "timestamp"]).reset_index(drop=True)

    data["Weekday"] = data["timestamp"].dt.weekday
    data["Month"] = data["timestamp"].dt.month
    data["Quarter"] = data["timestamp"].dt.quarter
    data["day_number"] = (data["timestamp"] - data["timestamp"].min()).dt.days
    data["trend_sq"] = data["day_number"] ** 2

    data["h3_cell_enc"] = le.fit_transform(data["h3_cell"])

    # 🚨 TRAINING ONLY — safe to drop NaNs here
    data = data.dropna().reset_index(drop=True)

    feature_columns = [
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "isHoliday",
        "neighbor_availability",
        "h3_cell_enc",
        "Weekday",
        "Month",
        "Quarter",
        "day_number",
        "trend_sq",
    ]

    X = data[feature_columns]
    y = data["demand"]

    return data, X, y, feature_columns


# ------------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------------
def train_model():
    data = export_data_from_mongo()
    data, X, y, feature_columns = prepare_training_features(data)
    train_raw, val_raw, val_true = per_h3_time_split(data)

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    X_train = train_raw[feature_columns]
    y_train = train_raw["demand"]
    X_val = val_raw[feature_columns]
    y_val = val_true["demand"]

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    print("✅ Validation R²:", r2_score(y_val, preds))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": feature_columns,
                "encoder": le,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
        )

    print(f"💾 Model saved → {MODEL_PATH}")
    return model


# ------------------------------------------------------------------
# INFERENCE FEATURE BUILDER
# ------------------------------------------------------------------
def build_inference_features(doc, encoder, feature_columns):
    df = pd.DataFrame([doc])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["Weekday"] = df["timestamp"].dt.weekday
    df["Month"] = df["timestamp"].dt.month
    df["Quarter"] = df["timestamp"].dt.quarter

    # Neutral trend values for inference
    df["day_number"] = 0
    df["trend_sq"] = 0

    # Encode H3
    df["h3_cell_enc"] = encoder.transform(df["h3_cell"])

    # Safe defaults
    df["neighbor_availability"] = df.get("neighbor_availability", 1.0)

    return df[feature_columns]


# ------------------------------------------------------------------
# MODEL LOADING (Call once at startup)
# ------------------------------------------------------------------
def load_model():
    """Load model from disk once at startup. Returns model_data dict."""
    print(f"📦 Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print("✅ Model loaded successfully!")
    return model_data


# ------------------------------------------------------------------
# DEMAND + PRICING PREDICTION (using pre-loaded model)
# ------------------------------------------------------------------
def predict_demand_with_model(h3_cell, timestamp, model_data):
    """
    Predict demand factor using pre-loaded model.
    
    Args:
        h3_cell: H3 cell identifier
        timestamp: Prediction timestamp
        model_data: Pre-loaded model dict from load_model()
    
    Returns:
        float: Demand factor (1.0 - 2.0)
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)

    model = model_data["model"]
    encoder = model_data["encoder"]
    feature_columns = model_data["features"]

    with get_db_connection() as db:
        doc = db[COLLECTION].find_one({
            "h3_cell": h3_cell,
            "timestamp": timestamp
        })

        if not doc:
            return None

        X = build_inference_features(doc, encoder, feature_columns)

        predicted_demand = float(model.predict(X)[0])

        capacity = max(doc.get("total_capacity", 1), 1)
        availability_ratio = doc.get("availability_ratio", 1)

        demand_pressure = predicted_demand / capacity
        scarcity = max(0, 1 - availability_ratio)

        demand_factor = min(max(
            1 + 0.6 * demand_pressure + 0.4 * scarcity,
            1.0
        ), 2.0)

        return demand_factor


# ------------------------------------------------------------------
# LEGACY FUNCTION (loads model each time - for backwards compatibility)
# ------------------------------------------------------------------
def predict_demand(h3_cell, timestamp):
    """
    Legacy function that loads model on each call.
    Use predict_demand_with_model() with pre-loaded model for better performance.
    """
    model_data = load_model()
    return predict_demand_with_model(h3_cell, timestamp, model_data)


# ------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
