---
license: mit
tags:
  - xgboost
  - parking
  - demand-prediction
  - time-series
  - dynamic-pricing
library_name: xgboost
---

# Dynamic Parking Demand Prediction Model

## Model Description
This XGBoost model predicts parking demand factors based on location (H3 cells) and temporal features. It's designed for dynamic pricing systems that adjust parking rates based on predicted demand.

## Model Details
- **Model Type**: XGBoost Regressor
- **Task**: Demand Factor Prediction (range: 1.0 - 2.0)
- **Framework**: XGBoost 2.0.0
- **Python Version**: 3.11
- **Training Date**: 2026

### Features Used
The model uses the following features for prediction:
- **Temporal Features**: 
  - `hour_sin`, `hour_cos`: Cyclical hour encoding
  - `Weekday`: Day of week (0-6)
  - `Month`: Month of year (1-12)
  - `Quarter`: Quarter of year (1-4)
  - `is_weekend`: Weekend indicator
  - `isHoliday`: Holiday indicator
  
- **Spatial Features**:
  - `h3_cell_enc`: Encoded H3 cell identifier
  - `neighbor_availability`: Parking availability in neighboring areas

- **Trend Features**:
  - `day_number`: Days since data start
  - `trend_sq`: Squared trend for non-linear patterns

## Training Data
- **Data Source**: Hourly parking occupancy data
- **Time Range**: 1 year of historical data
- **Spatial Coverage**: Multiple H3 cells (5km resolution)
- **Granularity**: Hourly observations

## Model Architecture
```python
XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)
```

## Intended Use
This model is designed to:
- Predict parking demand factors for dynamic pricing systems
- Adjust parking rates based on predicted demand (1.0x to 2.0x base price)
- Support real-time pricing decisions for parking management systems

### Primary Use Cases
- Smart city parking management
- Dynamic pricing optimization
- Parking availability forecasting
- Revenue optimization for parking operators

## How to Use

### Via API (Deployed on Render)
```bash
curl -X POST "https://your-app.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "h3_cell": "8c2a100d1a2bfff",
    "timestamp": "2026-01-20 14:00:00"
  }'
```

### Load Model Directly
```python
import pickle
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="your-username/dynamic-parking-demand-model",
    filename="demand_prediction_model.pkl"
)

# Load model
with open(model_path, "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    encoder = model_data["encoder"]
    features = model_data["features"]
```

## Performance
- **Validation R²**: [Add your validation score after training]
- **Test Set Performance**: [Add test metrics]

## Limitations
- Requires MongoDB connection for feature retrieval in production
- Limited to H3 cells present in training data
- Assumes hourly time granularity
- May not generalize to regions with significantly different parking patterns
- Requires historical data for accurate predictions

## Ethical Considerations
- **Fairness**: The model should be monitored to ensure it doesn't create pricing disparities that unfairly affect certain demographics
- **Transparency**: Dynamic pricing should be clearly communicated to end users
- **Accessibility**: Consider price caps to ensure parking remains accessible

## Training Details
- **Training Framework**: scikit-learn 1.3.0, XGBoost 2.0.0
- **Data Preprocessing**: Label encoding for categorical features, cyclical encoding for time
- **Validation Strategy**: Per-H3-cell time-based split (last 24 hours held out)

## Citation
```bibtex
@software{dynamic_parking_demand_2026,
  author = {Your Name},
  title = {Dynamic Parking Demand Prediction Model},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/your-username/dynamic-parking-demand-model}
}
```

## Contact
For questions or issues, please open an issue on the GitHub repository.
