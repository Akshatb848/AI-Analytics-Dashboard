# AI Analytics Dashboard — production image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8501

# Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --start-interval=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# ZAI_API_KEY (optional) and MAX_UPLOAD_ROWS are read from the environment at runtime
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
