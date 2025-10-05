FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    lsof \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN uv sync --frozen

# Create logs directory
RUN mkdir -p logs

# Make entrypoint script executable
RUN chmod +x scripts/docker-entrypoint.sh

# Expose ports
EXPOSE 4000 8082

# Health check (health endpoint doesn't require auth)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# Start both services
ENTRYPOINT ["scripts/docker-entrypoint.sh"]
