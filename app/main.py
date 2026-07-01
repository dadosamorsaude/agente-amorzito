from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.metrics import router as metrics_router
from app.core.logger import logger
import time
import os
from dotenv import load_dotenv

# Carrega arquivos .env pro os.environ (essencial pro LangSmith enxergar as chaves no ambiente)
load_dotenv(override=True)


app = FastAPI(
    title="AMORZITO AI Agent",
    version="0.1.0",
    description="Agente de análise de prontuários médicos.",
)

from app.core.config import settings

# Parse de domínios permitidos via variável de ambiente (separados por vírgula)
allowed_origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]

# If '*' is specified, we must set allow_credentials to False because FastAPI's CORSMiddleware
# raises an error if allow_origins=['*'] and allow_credentials=True.
# Since we authenticate using custom headers (X-API-Key/Authorization) and not browser cookies,
# it is safe and correct to disable credentials when using '*' to support all preview/local domains.
allow_all = "*" in allowed_origins or not allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else allowed_origins,
    allow_credentials=False if allow_all else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"Completed {request.method} {request.url.path} with status {response.status_code} in {process_time:.3f}s")

    return response


app.include_router(chat_router)
app.include_router(metrics_router)


@app.get("/")
def home():
    """Health check endpoint for Render monitoring."""
    logger.info("Health check endpoint called.")
    return {
        "status": "ok",
        "agent": "AMORZITO",
        "version": "0.1.0",
        "environment": "production" if os.getenv("RENDER") else "development"
    }