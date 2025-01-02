FROM python:3.9-slim

RUN pip install requests

WORKDIR /app

COPY client.py .

ENV CSV_PATH=/app/data/songs.csv
ENV API_URL=http://10.43.62.138:52043/api/recommend
ENV OUTPUT_PATH=/app/output/responses.txt
ENV NUM_REQUESTS=10

RUN mkdir -p /app/data /app/output

CMD ["python", "client.py"]
