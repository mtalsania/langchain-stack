# Free-host friendly image for module 12 (Render, Fly.io, Hugging Face Spaces, Railway trial...)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Download the embedding model and build the index at build time, so cold starts are fast.
RUN python -c "from common.retrieval import build_index; build_index(rebuild=True)"
ENV PORT=8000
CMD uvicorn 12_server:app --host 0.0.0.0 --port ${PORT}
