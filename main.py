# main.py
import uvicorn
from fastapi import FastAPI
from services.patrol_api import router as patrol_router


app = FastAPI(title="Patrol Planning API")
app.include_router(patrol_router)


if __name__ == "__main__":
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
