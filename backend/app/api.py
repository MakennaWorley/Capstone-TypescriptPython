from typing import Any, Dict
from fastapi.responses import JSONResponse

def api_success(message: str, data: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "success", "message": message, "data": data},
    )


def api_error(message: str, status_code: int, code: str = "error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "code": code, "message": message},
    )