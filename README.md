# GenAI API — Smart Resume Parser

## 1. Описание проекта

Production-ready API для генерации структурированных данных из текста резюме с помощью LLM-модели **Qwen2.5-7B-Instruct**. Приложение принимает текст резюме и возвращает структурированный JSON с извлечёнными полями (ФИО, контакты, образование, опыт, навыки) было трудно но зато какой проект. Этот проект является продолжением моего прошлого проекта который вы можете найти в репозиториях

Модель работает через HuggingFace Inference Providers API — веса не хранятся локально, инференс выполняется на серверах HuggingFace.

## 2. Архитектура

```
Пользователь → FastAPI (Python) → HuggingFace Inference API → Qwen2.5-7B-Instruct
                   ↓
              Docker-контейнер
                   ↓
              Railway (облако)
                   ↓
           GitHub Actions (CI/CD)
```

Пайплайн: `git push → lint → test → docker build → deploy → smoke test`

## 3. Переменные окружения

| Переменная     | Назначение                                   | Обязательна |
|----------------|----------------------------------------------|-------------|
| `HF_TOKEN`     | Токен HuggingFace для Inference Providers API | Да          |
| `MODEL_NAME`   | ID модели на HuggingFace Hub                 | Нет (default: `Qwen/Qwen2.5-7B-Instruct`) |
| `MAX_NEW_TOKENS` | Максимум токенов по умолчанию              | Нет (default: `256`) |
| `PORT`         | Порт сервера                                  | Нет (default: `8000`) |

## 4. Локальный запуск (без Docker)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Создайте .env файл с HF_TOKEN
echo "HF_TOKEN=hf_your_token" > .env

uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

## 5. Запуск через Docker

```bash
# Сборка образа
docker build -t genai-api .

# Запуск контейнера
docker run -p 8000:8000 -e HF_TOKEN=hf_your_token genai-api
```

## 6. Ссылка на деплой

> **Публичный URL**: `https://smartresumeparserdeploy-production.up.railway.app`
>
> *(обновите после деплоя на Railway)*

## 7. Описание CI/CD

Пайплайн запускается автоматически при каждом `push` в ветку `main`.

Шаги выполняются последовательно через `needs`:

1. **Lint** — проверка кода линтером `ruff`
2. **Test** — запуск `pytest` (проверка `/health`, `/generate`, валидация)
3. **Build** — сборка Docker-образа для проверки Dockerfile
4. **Deploy** — развертывание на Railway
5. **Smoke test** — проверка что `/health` и `/` отвечают после деплоя

Если любой шаг падает — последующие не выполняются.

## 8. Пример запроса/ответа

### Health check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### Информация о сервисе
```bash
curl http://localhost:8000/
```
```json
{"service": "GenAI API", "version": "1.0.0", "description": "Fine-tuned LLM inference API"}
```

### Генерация (валидный запрос)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "John Doe\njohn@email.com\n(555) 123-4567\nSkills: Python, Go", "max_tokens": 512}'
```
```json
{
  "prompt": "John Doe\njohn@email.com\n(555) 123-4567\nSkills: Python, Go",
  "response": "{\"name\": \"John Doe\", \"email\": \"john@email.com\", ...}",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "tokens_used": 142
}
```

### Невалидный запрос (пустой промпт → 422)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": ""}'
```
```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

## 9. Известные ограничения

- **Модель работает через API** — зависимость от доступности HuggingFace Inference Providers
- **Холодный старт** — первый запрос после простоя может занять 10–30 секунд (модель загружается на серверах HF)
- **Бесплатный тариф HuggingFace** — ограничение по количеству запросов (~200/час)
- **Railway Free Tier** — ограничение $5/месяц, сервис может засыпать при неактивности
- **Без локальных весов** — адаптеры QLoRA из Проекта 21 не используются при облачном инференсе
