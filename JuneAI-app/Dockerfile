# JuneAI — Streamlit frontend + LangGraph agent
# Build: docker build -t juneai .
# Run:   docker compose up

FROM python:3.11-slim

# System deps for Pillow and SQLite (already bundled in slim but kept explicit)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so layer is cached when code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check — Streamlit serves HTTP on 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit config: disable telemetry, run headless
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
