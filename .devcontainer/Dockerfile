FROM python:3.11-slim-bookworm

RUN set -e && \
    # Install base OS packages
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        git && \
    \
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ## Add uv on PATH
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    \
    ## Add Google Cloud SDK APT repo & key
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    \
    # Install gcloud CLI
    apt-get update && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    \
    # Clean up
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/GEST

# Set environment variables for gcloud
ENV CLOUDSDK_CONFIG=/workspaces/GEST/.secrets/gcloud   \
    GOOGLE_APPLICATION_CREDENTIALS=/workspaces/GEST/.secrets/gcloud/application_default_credentials.json

# Ensure the directory for gcloud config exists
RUN mkdir -p "$CLOUDSDK_CONFIG"

# Ensure packages are installed
COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/usr/local

RUN uv sync --frozen --no-dev --extra test --extra develop