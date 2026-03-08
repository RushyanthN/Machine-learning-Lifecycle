from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import clean, visualize, transform, predict, eda, features

app = FastAPI(title="ML Prediction Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(clean.router, prefix="/api")
app.include_router(eda.router, prefix="/api")
app.include_router(visualize.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(transform.router, prefix="/api")
app.include_router(predict.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "ML Prediction Platform API v2"}
