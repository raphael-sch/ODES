FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn
RUN pip install gevent gevent-websocket

# Default command (can be overridden by docker-compose)
CMD ["python", "app.py"]
