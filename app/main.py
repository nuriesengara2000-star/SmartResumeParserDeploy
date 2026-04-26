"""Точка входа FastAPI-приложения GenAI API."""

import logging
import signal
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import inference_engine
from app.models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ServiceInfoResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузка модели при старте и очистка при завершении."""
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
    description=(
        "Production API для генерации текста с помощью дообученной LLM. "
        "Контейнеризовано и развёрнуто через CI/CD."
    ),
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


# --- Graceful shutdown ---
def handle_sigterm(*_args) -> None:
    """Корректная обработка SIGTERM от Docker/Railway."""
    logger.info("Получен SIGTERM, завершаю работу…")
    raise SystemExit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


# --- Эндпоинты ---


@app.get(
    "/",
    response_model=ServiceInfoResponse,
    summary="Информация о сервисе",
)
async def root() -> ServiceInfoResponse:
    """Возвращает название, версию и описание сервиса."""
    return ServiceInfoResponse(
        service="GenAI API",
        version=SERVICE_VERSION,
        description="Fine-tuned LLM inference API",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка работоспособности",
)
async def health() -> HealthResponse:
    """Health check для Docker и облачной платформы."""
    return HealthResponse(status="ok")


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
    """Принимает промпт и возвращает сгенерированный текст."""
    if not inference_engine.is_loaded:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        result = await inference_engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        logger.exception("Ошибка при генерации: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка модели: {exc}",
        )

    return GenerateResponse(
        prompt=request.prompt,
        response=result["response"],
        model=result["model"],
        tokens_used=result["tokens_used"],
    )
