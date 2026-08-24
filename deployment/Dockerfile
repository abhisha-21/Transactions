FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.api:app"]
