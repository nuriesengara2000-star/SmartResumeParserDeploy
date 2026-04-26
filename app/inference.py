"""Логика инференса через HuggingFace Inference Providers API."""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

SYSTEM_INSTRUCTION = (
    "Extract structured information from the resume and return only valid JSON."
)


class ModelInference:
    """Инференс через HuggingFace Inference Providers API."""

    def __init__(self) -> None:
        self._loaded = False
        self._client: httpx.AsyncClient | None = None

    def load(self) -> None:
        """Инициализация HTTP-клиента при старте."""
        logger.info("Инициализация HF Inference API клиента…")

        if not settings.HF_TOKEN:
            raise RuntimeError("HF_TOKEN не задан в переменных окружения")

        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {settings.HF_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        self._loaded = True
        logger.info("HF Inference API готов (%s)", settings.MODEL_NAME)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return settings.MODEL_NAME

    async def generate(self, prompt: str, max_tokens: int = 256) -> dict:
        """Отправляет запрос к HuggingFace Inference Providers API."""
        payload = {
            "model": settings.MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.01,
        }

        response = await self._client.post(HF_API_URL, json=payload)

        if response.status_code != 200:
            logger.error(
                "HF API ошибка %s: %s", response.status_code, response.text[:500]
            )
            response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        tokens_used = data.get("usage", {}).get("total_tokens", 0)

        return {
            "response": content,
            "model": settings.MODEL_NAME,
            "tokens_used": tokens_used,
        }

    async def close(self) -> None:
        """Закрытие HTTP-клиента."""
        if self._client:
            await self._client.aclose()


inference_engine = ModelInference()


