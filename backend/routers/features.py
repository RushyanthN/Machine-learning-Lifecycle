from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from backend.utils import get_dataframe, update_dataframe, _sanitize, dataframe_summary
from backend.services.features import (
    get_correlation_matrix, get_covariance_matrix, get_vif,
    get_feature_importance, create_column, delete_columns,
)

router = APIRouter(tags=["features"])


class SessionRequest(BaseModel):
    session_id: str


class FeatureImportanceRequest(BaseModel):
    session_id: str
    target_col: str
    task: str = "regression"


class CreateColumnRequest(BaseModel):
    session_id: str
    name: str
    expression: str


class DeleteColumnsRequest(BaseModel):
    session_id: str
    columns: List[str]


@router.post("/features/correlation")
async def correlation(req: SessionRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_correlation_matrix(df))


@router.post("/features/covariance")
async def covariance(req: SessionRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_covariance_matrix(df))


@router.post("/features/vif")
async def vif(req: SessionRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_vif(df))


@router.post("/features/importance")
async def importance(req: FeatureImportanceRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_feature_importance(df, req.target_col, req.task))


@router.post("/features/create")
async def create_col(req: CreateColumnRequest):
    df = get_dataframe(req.session_id)
    df = create_column(df, req.name, req.expression)
    update_dataframe(req.session_id, df)
    return {"summary": _sanitize(dataframe_summary(df))}


@router.post("/features/delete")
async def delete_cols(req: DeleteColumnsRequest):
    df = get_dataframe(req.session_id)
    df = delete_columns(df, req.columns)
    update_dataframe(req.session_id, df)
    return {"summary": _sanitize(dataframe_summary(df))}
