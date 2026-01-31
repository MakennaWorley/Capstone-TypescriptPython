import sys
import os
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from data_generation import create_data_from_params

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

# health check
@app.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI"}

# Create Dataset
@app.post("/api/create/data")
async def create_dataset(request: Request):
    # Print a specific header
    x_api_key = request.headers.get("x-api-key")
    print("X-API-Key:", x_api_key)

    # Print only headers you care about
    interesting = ["x-api-key", "content-type", "origin", "referer"]
    print("=== Selected Headers ===")
    for h in interesting:
        print(f"{h}: {request.headers.get(h)}")

    # Print the JSON body
    body = await request.json()
    print("=== Body ===")
    print(body)

    try:
        simulation_params = body.get("params", body)
        
        os.makedirs(DATASETS_DIR, exist_ok=True)
        
        simulation_params["output_dir"] = str(DATASETS_DIR)

        result = create_data_from_params(simulation_params)

        return {
            "status": "success",
            "message": "Data generated successfully",
            "details": {
                "config_used": result["config"],
                "file_paths": result["outputs"]
            },
            "data": result
        }

    except Exception as e:
        print(f"Error during data generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))