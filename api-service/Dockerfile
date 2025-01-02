FROM python:3.10-slim

WORKDIR /app

RUN pip install flask gunicorn

COPY app.py .

RUN mkdir -p /app/shared/models

ENV API_VERSION=1.0
ENV MODEL_PATH=/app/shared/models/fpgrowth_model.pkl
ENV PORT=52043
ENV CHECK_INTERVAL=30

EXPOSE 52043

CMD ["gunicorn", "--bind", "0.0.0.0:52043", "app:app"]