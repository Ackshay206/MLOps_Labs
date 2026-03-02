from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class BreastCancerData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

class BreastCancerResponse(BaseModel):
    prediction: int
    diagnosis: str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=BreastCancerResponse)
async def predict_cancer(cancer_features: BreastCancerData):
    try:
        features = [[
            cancer_features.mean_radius, cancer_features.mean_texture,
            cancer_features.mean_perimeter, cancer_features.mean_area,
            cancer_features.mean_smoothness, cancer_features.mean_compactness,
            cancer_features.mean_concavity, cancer_features.mean_concave_points,
            cancer_features.mean_symmetry, cancer_features.mean_fractal_dimension,
            cancer_features.radius_error, cancer_features.texture_error,
            cancer_features.perimeter_error, cancer_features.area_error,
            cancer_features.smoothness_error, cancer_features.compactness_error,
            cancer_features.concavity_error, cancer_features.concave_points_error,
            cancer_features.symmetry_error, cancer_features.fractal_dimension_error,
            cancer_features.worst_radius, cancer_features.worst_texture,
            cancer_features.worst_perimeter, cancer_features.worst_area,
            cancer_features.worst_smoothness, cancer_features.worst_compactness,
            cancer_features.worst_concavity, cancer_features.worst_concave_points,
            cancer_features.worst_symmetry, cancer_features.worst_fractal_dimension
        ]]

        prediction = predict_data(features)
        diagnosis = "benign" if prediction[0] == 1 else "malignant"
        return BreastCancerResponse(prediction=int(prediction[0]), diagnosis=diagnosis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
