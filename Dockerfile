# ════════════════════════════════════════════════════════
#  Stage 1: Builder (Compiler & Dependency Installer)
# ════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY req.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r req.txt

# ════════════════════════════════════════════════════════
#  Stage 2: Final Production Runtime Environment
# ════════════════════════════════════════════════════════
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime system packages and Docker CLI for Docker-out-of-Docker (DooD)
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set active path to builder virtual env
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy project source files
COPY . .

# Expose Telegram webhook / development ports if any
EXPOSE 8080

# Execute bot launch script
CMD ["python", "main.py"]
