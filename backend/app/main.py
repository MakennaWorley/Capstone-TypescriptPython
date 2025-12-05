import os
import io
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path

app = FastAPI()

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

@app.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI"}

@app.get("/poc/data/{file_name}/csv")
def get_output_csv(file_name: str):
    csv_path = DATASETS_DIR / f"{file_name}.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{csv_path.name}' not found in '{DATASETS_DIR}' inside the container.",
        )

    df = pd.read_csv(csv_path)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={file_name}.csv"},
    )