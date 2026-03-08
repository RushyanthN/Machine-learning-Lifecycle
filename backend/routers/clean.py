from fastapi import APIRouter, UploadFile, File, Form
from backend.utils import parse_csv, store_dataframe, get_dataframe, update_dataframe, dataframe_summary
from backend.services.cleaner import clean_data, detect_column_types

router = APIRouter(tags=["cleaning"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = parse_csv(contents)
    session_id = store_dataframe(df)
    col_types = detect_column_types(df)
    return {
        "session_id": session_id,
        "filename": file.filename,
        "summary": dataframe_summary(df),
        "column_types": col_types,
    }


@router.post("/clean")
async def clean_endpoint(
    session_id: str = Form(...),
    missing_strategy: str = Form("drop"),
    remove_duplicates: bool = Form(True),
):
    df = get_dataframe(session_id)
    cleaned_df, report = clean_data(df, missing_strategy, remove_duplicates)
    update_dataframe(session_id, cleaned_df)
    return {
        "session_id": session_id,
        "report": report,
        "summary": dataframe_summary(cleaned_df),
    }
