"""Pydantic-схемы запросов и ответов для GenAI API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Схема входящего запроса на генерацию."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Текст промпта для генерации",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Максимальное количество токенов в ответе",
    )


class GenerateResponse(BaseModel):
    """Схема ответа с результатом генерации."""

    prompt: str = Field(..., description="Текст отправленного промпта")
    response: str = Field(..., description="Сгенерированный текст модели")
    model: str = Field(..., description="Название использованной модели")
    tokens_used: int = Field(..., description="Количество использованных токенов")


class HealthResponse(BaseModel):
    """Схема ответа health-check эндпоинта."""

    status: str = Field(default="ok", description="Статус сервиса")


class ServiceInfoResponse(BaseModel):
    """Схема ответа корневого эндпоинта."""

    service: str = Field(..., description="Название сервиса")
    version: str = Field(..., description="Версия API")
    description: str = Field(..., description="Описание сервиса")


class ErrorResponse(BaseModel):
    """Схема ответа при ошибке."""

    detail: str = Field(..., description="Описание ошибки")
