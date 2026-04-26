"""Скрипт загрузки весов модели из HuggingFace Hub.

Используется при размере весов > 500 МБ.
В нашем случае модель загружается через HF Inference API,
поэтому локальные веса не требуются. Скрипт оставлен как
пример для моделей, требующих локальной загрузки.
"""

import os
import sys


def download_weights() -> None:
    """Загрузка весов модели из HuggingFace Hub."""
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    print(f"Модель {model_name} используется через HF Inference API.")
    print("Локальная загрузка весов не требуется.")


if __name__ == "__main__":
    download_weights()
