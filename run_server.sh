#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
PORT=8000
HOST="0.0.0.0"

echo "=================================================="
echo " Starting MediStock PaddleOCR + FastAPI Server"
echo " Host: $HOST, Port: $PORT"
echo " Endpoint: http://localhost:$PORT/ocr/parse-bill"
echo " Docs:     http://localhost:$PORT/docs"
echo "=================================================="

exec "$DIR/.venv/bin/uvicorn" app.main:app --host "$HOST" --port "$PORT" --reload
