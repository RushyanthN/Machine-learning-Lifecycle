from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.utils import get_dataframe, _sanitize
from backend.services.modeler import (
    train_and_evaluate, get_available_models,
    get_hyperparam_options, get_default_hyperparams,
)

router = APIRouter(tags=["prediction"])


class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    session_id: str
    target_col: str
    feature_cols: List[str]
    task: str
    model_configs: List[ModelConfig]
    test_size: float = 0.2


@router.get("/predict/models")
async def list_models(task: str = "regression"):
    return {"task": task, "models": get_available_models(task)}


@router.get("/predict/hyperparams")
async def hyperparams(model_name: str):
    return {
        "model": model_name,
        "options": get_hyperparam_options(model_name),
        "defaults": get_default_hyperparams(model_name),
    }


@router.post("/predict/train")
async def train_models(req: PredictRequest):
    df = get_dataframe(req.session_id)
    configs = [{"name": mc.name, "params": mc.params} for mc in req.model_configs]
    result = train_and_evaluate(
        df,
        target_col=req.target_col,
        feature_cols=req.feature_cols,
        task=req.task,
        model_configs=configs,
        test_size=req.test_size,
    )
    return _sanitize(result)
