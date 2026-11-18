
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Shrimp Oxygen Risk Prediction API",
    description="Predicts night-time oxygen depletion risk"
)

model = mlflow.sklearn.load_model(
    "models:/ShrimpOxygenRisk@champion"
)

FEATURES = [
    "pond_area_acres",
    "stocking_density_shrimp_per_m2",
    "oxygen_8pm_mg_L",
    "water_temperature_C",
    "daily_feed_kg",
    "planned_aerator_hours",
    "expected_rainfall_mm",
    "feed_per_unit_biomass",
    "aeration_hours_per_area"
]

class PondInput(BaseModel):

    pond_area_acres: float = Field(gt=0)
    stocking_density_shrimp_per_m2: float = Field(gt=0)
    oxygen_8pm_mg_L: float = Field(ge=0)
    water_temperature_C: float = Field(gt=0, le=50)
    daily_feed_kg: float = Field(ge=0)
    planned_aerator_hours: float = Field(ge=0, le=12)
    expected_rainfall_mm: float = Field(ge=0)


@app.get("/health")
def health():

    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PondInput):

    values = pd.DataFrame([data.model_dump()])

    biomass = (
        values["pond_area_acres"] *
        4046.86 *
        values["stocking_density_shrimp_per_m2"] *
        0.02
    )

    values["feed_per_unit_biomass"] = (
        values["daily_feed_kg"] / biomass
    )

    values["aeration_hours_per_area"] = (
        values["planned_aerator_hours"] /
        values["pond_area_acres"]
    )

    values = values[FEATURES]

    prediction = model.predict(values)[0]

    probabilities = model.predict_proba(values)[0]

    high_index = list(model.classes_).index("High")

    return {
        "oxygen_risk": str(prediction),
        "high_risk_probability": round(
            float(probabilities[high_index]), 4
        ),
        "additional_night_aeration": (
            "Required" if prediction == "High"
            else "Not indicated by model"
        )
    }
