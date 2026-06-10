FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "-c", \
    "python scripts/generate_transactions.py && \
     dbt build --profiles-dir . && \
     python scripts/predict.py && \
     python scripts/generate_report.py"]