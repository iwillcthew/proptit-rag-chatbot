FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory with proper permissions
RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache

COPY requirements.txt ./
COPY . ./

RUN pip3 install -r requirements.txt

# Pre-download the embedding model to avoid permission issues at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('iwillcthew/vietnamese-embedding-PROPTIT-domain-ft')"

# Read secrets from Docker secrets and set them as environment variables
RUN --mount=type=secret,id=NIM_API_KEY,mode=0444,required=true \
    --mount=type=secret,id=MONGODB_URI,mode=0444,required=true \
    echo "NIM_API_KEY=$(cat /run/secrets/NIM_API_KEY)" > /app/.env && \
    echo "MONGODB_URI=$(cat /run/secrets/MONGODB_URI)" >> /app/.env

# Optional: Add HF_TOKEN if provided
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        echo "HF_TOKEN=$(cat /run/secrets/HF_TOKEN)" >> /app/.env; \
    fi

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
