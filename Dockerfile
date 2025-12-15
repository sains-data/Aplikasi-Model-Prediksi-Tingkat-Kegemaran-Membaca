FROM python:3.11-slim

WORKDIR /app

# Install system deps (matplotlib + statsmodels)
RUN apt-get update && apt-get install -y \
    gcc g++ libfreetype6-dev libpng-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data models

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
