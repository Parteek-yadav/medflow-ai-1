from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="Multi-Agent AI Backend")
app.include_router(router, prefix="/api")
