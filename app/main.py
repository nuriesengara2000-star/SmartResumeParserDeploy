"""Точка входа FastAPI-приложения GenAI API."""

import logging
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

from app.inference import inference_engine
from app.models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ServiceInfoResponse,
)

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Запуск приложения — инициализация модели…")
    try:
        inference_engine.load()
        logger.info("Сервис готов к работе.")
    except Exception as exc:
        logger.error("Не удалось инициализировать модель: %s", exc)
        raise
    yield
    logger.info("Завершение приложения — очистка ресурсов…")
    await inference_engine.close()
    logger.info("Ресурсы освобождены.")


app = FastAPI(
    title="GenAI API",
    description="Production API для генерации текста с помощью дообученной LLM.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def handle_sigterm(*_args) -> None:
    logger.info("Получен SIGTERM, завершаю работу…")
    raise SystemExit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


# --- Эндпоинты ---

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/chat")


@app.get("/info", response_model=ServiceInfoResponse, summary="Информация о сервисе")
async def info() -> ServiceInfoResponse:
    return ServiceInfoResponse(
        service="GenAI API",
        version=SERVICE_VERSION,
        description="Fine-tuned LLM inference API",
    )


@app.get("/health", response_model=HealthResponse, summary="Проверка работоспособности")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/chat", response_class=HTMLResponse, summary="Веб-интерфейс чата")
async def chat_ui() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Генерация текста по промпту",
    responses={
        422: {"description": "Невалидные входные данные"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"},
    },
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    if not inference_engine.is_loaded:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    try:
        result = await inference_engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        logger.exception("Ошибка при генерации: %s", exc)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка модели: {exc}")

    return GenerateResponse(
        prompt=request.prompt,
        response=result["response"],
        model=result["model"],
        tokens_used=result["tokens_used"],
    )

