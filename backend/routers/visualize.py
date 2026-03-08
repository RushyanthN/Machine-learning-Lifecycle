from fastapi import APIRouter, Form
from pydantic import BaseModel
from typing import List, Optional
from backend.utils import get_dataframe
from backend.services.visualizer import recommend_plots, generate_plot

router = APIRouter(tags=["visualization"])


class PlotRequest(BaseModel):
    session_id: str
    plot_type: str
    columns: List[str]
    title: Optional[str] = ""


@router.post("/visualize/recommend")
async def get_recommendations(session_id: str = Form(...)):
    df = get_dataframe(session_id)
    recs = recommend_plots(df)
    return {"session_id": session_id, "recommendations": recs}


@router.post("/visualize/plot")
async def create_plot(req: PlotRequest):
    df = get_dataframe(req.session_id)
    fig_dict = generate_plot(df, req.plot_type, req.columns, req.title or "")
    return {"plot": fig_dict}
