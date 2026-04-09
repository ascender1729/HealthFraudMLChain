FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV RESULTS_DIR=/app/results
EXPOSE 5000

CMD ["python", "app/app.py"]
