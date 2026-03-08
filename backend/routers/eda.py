from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.utils import get_dataframe, _sanitize
from backend.services.eda import (
    get_data_overview, get_info, get_describe,
    get_null_analysis, get_value_counts,
)

router = APIRouter(tags=["eda"])


class SessionRequest(BaseModel):
    session_id: str


class ValueCountsRequest(BaseModel):
    session_id: str
    column: str
    top_n: int = 20


@router.post("/eda/overview")
async def overview(req: SessionRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_data_overview(df))


@router.post("/eda/info")
async def info(req: SessionRequest):
    df = get_dataframe(req.session_id)
    return _sanitize({"info": get_info(df)})


@router.post("/eda/describe")
async def describe(req: SessionRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_describe(df))


@router.post("/eda/nulls")
async def null_analysis(req: SessionRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_null_analysis(df))


@router.post("/eda/value_counts")
async def value_counts(req: ValueCountsRequest):
    df = get_dataframe(req.session_id)
    return _sanitize(get_value_counts(df, req.column, req.top_n))
