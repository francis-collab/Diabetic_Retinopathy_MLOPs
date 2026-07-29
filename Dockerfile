# Official TF 2.13 image – TensorFlow is already installed
FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Lightweight requirements (NO tensorflow – already in the base image)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-docker.txt

# Project code
COPY api/          ./api/
COPY app/          ./app/
COPY src/          ./src/
COPY models/       ./models/
COPY data/         ./data/
COPY scripts/      ./scripts/

# locust folder is optional – only copy if it exists on the host
# (create an empty locust/ dir first if you don't have one yet)
COPY locust/       ./locust/

ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]