from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from backend.utils import get_dataframe, update_dataframe, dataframe_summary
from backend.services.transformer import (
    apply_scaling,
    apply_encoding,
    apply_log_transform,
    apply_polynomial_features,
    get_transform_preview,
)

router = APIRouter(tags=["transformation"])


class ScaleRequest(BaseModel):
    session_id: str
    columns: List[str]
    method: str = "standard"


class EncodeRequest(BaseModel):
    session_id: str
    columns: List[str]
    method: str = "onehot"


class LogTransformRequest(BaseModel):
    session_id: str
    columns: List[str]


class PolyRequest(BaseModel):
    session_id: str
    columns: List[str]
    degree: int = 2


class PreviewRequest(BaseModel):
    session_id: str


@router.post("/transform/preview")
async def transform_preview(req: PreviewRequest):
    df = get_dataframe(req.session_id)
    return get_transform_preview(df)


@router.post("/transform/scale")
async def scale_endpoint(req: ScaleRequest):
    df = get_dataframe(req.session_id)
    df, info = apply_scaling(df, req.columns, req.method)
    update_dataframe(req.session_id, df)
    return {"info": info, "summary": dataframe_summary(df)}


@router.post("/transform/encode")
async def encode_endpoint(req: EncodeRequest):
    df = get_dataframe(req.session_id)
    df, info = apply_encoding(df, req.columns, req.method)
    update_dataframe(req.session_id, df)
    return {"info": info, "summary": dataframe_summary(df)}


@router.post("/transform/log")
async def log_transform_endpoint(req: LogTransformRequest):
    df = get_dataframe(req.session_id)
    df, info = apply_log_transform(df, req.columns)
    update_dataframe(req.session_id, df)
    return {"info": info, "summary": dataframe_summary(df)}


@router.post("/transform/polynomial")
async def polynomial_endpoint(req: PolyRequest):
    df = get_dataframe(req.session_id)
    df, info = apply_polynomial_features(df, req.columns, req.degree)
    update_dataframe(req.session_id, df)
    return {"info": info, "summary": dataframe_summary(df)}
