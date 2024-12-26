FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY recommendation.py .

RUN mkdir -p /app/shared/models

ENV MODEL_PATH=/app/shared/models/fpgrowth_model.pkl
ENV DATASET_PATH=/data/2023_spotify_ds1.csv

CMD ["python", "recommendation.py", "/data/2023_spotify_ds1.csv"]