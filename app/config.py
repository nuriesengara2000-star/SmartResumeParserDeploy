"""Конфигурация приложения через переменные окружения."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из переменных окружения."""

    MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    MAX_NEW_TOKENS: int = 256
    HF_TOKEN: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
