"""Тесты API эндпоинтов."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Устанавливаем HF_TOKEN до импорта приложения
os.environ.setdefault("HF_TOKEN", "hf_test_token_for_ci")

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async HTTP клиент для тестирования."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
@pytest.mark.anyio
async def test_root(client: AsyncClient) -> None:
    """Тест корневого эндпоинта / — редирект на /chat."""
    response = await client.get("/", follow_redirects=False)
    assert response.status_code == 307


@pytest.mark.anyio
async def test_health(client: AsyncClient) -> None:
    """Тест /health — должен вернуть 200 и status ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_generate_valid(client: AsyncClient) -> None:
    """Тест /generate с валидным запросом."""
    mock_result = {
        "response": '{"name": "John Doe"}',
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "tokens_used": 42,
    }

    with patch.object(
        app.state if hasattr(app, "state") else app,
        "__dict__",
        {},
    ):
        from app.inference import inference_engine

        inference_engine._loaded = True
        with patch.object(
            inference_engine, "generate", new_callable=AsyncMock, return_value=mock_result
        ):
            response = await client.post(
                "/generate",
                json={"prompt": "Parse this resume: John Doe"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "prompt" in data
            assert "response" in data
            assert "model" in data
            assert "tokens_used" in data
            assert data["prompt"] == "Parse this resume: John Doe"


@pytest.mark.anyio
async def test_generate_empty_prompt(client: AsyncClient) -> None:
    """Тест /generate с пустым промптом — должен вернуть 422."""
    response = await client.post(
        "/generate",
        json={"prompt": ""},
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_generate_prompt_too_long(client: AsyncClient) -> None:
    """Тест /generate с промптом > 4096 символов — должен вернуть 422."""
    response = await client.post(
        "/generate",
        json={"prompt": "x" * 4097},
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_generate_invalid_max_tokens(client: AsyncClient) -> None:
    """Тест /generate с невалидным max_tokens — должен вернуть 422."""
    response = await client.post(
        "/generate",
        json={"prompt": "Hello", "max_tokens": 9999},
    )
    assert response.status_code == 422

@pytest.mark.anyio
async def test_info(client: AsyncClient) -> None:
    """Тест /info — информация о сервисе."""
    response = await client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "GenAI API"
    assert "version" in data
    assert "description" in data

