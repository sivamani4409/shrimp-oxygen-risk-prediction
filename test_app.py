import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# 1. Data Validation Tests
def test_dataset_properties():
    df = pd.read_csv("shrimp_oxygen_dataset_v1_v2.csv")
    
    # Ensure critical columns exist
    required_cols = [
        "pond_area_acres", "stocking_density_shrimp_per_m2", 
        "oxygen_8pm_mg_L", "water_temperature_C", 
        "daily_feed_kg", "planned_aerator_hours", 
        "expected_rainfall_mm", "oxygen_risk"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"
        
    # Ensure data constraints are valid
    assert (df["pond_area_acres"] > 0).all()
    assert (df["stocking_density_shrimp_per_m2"] > 0).all()
    assert (df["water_temperature_C"] <= 50).all()
    assert df["oxygen_risk"].isin(["High", "Normal"]).all()

# 2. FastAPI Endpoint Tests
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_high_risk():
    payload = {
        "pond_area_acres": 1.0,
        "stocking_density_shrimp_per_m2": 150.0,
        "oxygen_8pm_mg_L": 3.0,
        "water_temperature_C": 30.0,
        "daily_feed_kg": 180.0,
        "planned_aerator_hours": 1.0,
        "expected_rainfall_mm": 20.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "oxygen_risk" in data
    assert "high_risk_probability" in data
    assert "additional_night_aeration" in data

def test_predict_validation_error():
    # Invalid temperature (> 50)
    invalid_payload = {
        "pond_area_acres": 1.0,
        "stocking_density_shrimp_per_m2": 100.0,
        "oxygen_8pm_mg_L": 6.0,
        "water_temperature_C": 55.0,
        "daily_feed_kg": 50.0,
        "planned_aerator_hours": 6.0,
        "expected_rainfall_mm": 0.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
