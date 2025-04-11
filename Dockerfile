FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY app/model /app/app/model 

EXPOSE 5000
CMD ["python", "app/main.py"]
