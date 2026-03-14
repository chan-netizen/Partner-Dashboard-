# Partner Readiness & Training Dashboard

A Streamlit dashboard for partner onboarding, training, certification readiness, performance, and risk monitoring.

## Files
- `app.py` - Main Streamlit application
- `partner_training_dashboard_data.csv` - Source dataset
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Optional Streamlit theme settings

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard sections
- Overview
- Training
- Performance
- Predictive
- Actions

## Predictive models
- Random forest classifier for at-risk partners
- Random forest regressor for performance score prediction
- K-means segmentation for partner personas
