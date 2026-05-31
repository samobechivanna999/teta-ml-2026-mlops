FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/logs /app/input /app/output && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs /app/input /app/output

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/build_model.py

VOLUME /app/input
VOLUME /app/output

CMD ["python", "./app/app.py"]
