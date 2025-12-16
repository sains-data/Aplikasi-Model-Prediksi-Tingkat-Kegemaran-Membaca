FROM python:3.11-slim

WORKDIR /app

# System dependencies untuk numpy / statsmodels / matplotlib
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pastikan folder ada
RUN mkdir -p data models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
