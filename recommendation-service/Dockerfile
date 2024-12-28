FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY recommendation.py .

RUN mkdir -p /app/shared/models

ENV MODEL_PATH=/app/shared/models/fpgrowth_model.pkl
ENV CENTRAL_DATASET_PATH=/central_data/2023_spotify_ds1.csv
ENV USER_DATASET_PATH=/user_data/2023_spotify_songs.csv
ENV CHECK_INTERVAL=30

CMD ["python", "recommendation.py"]
