FROM nvidia/cuda:12.5.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory and copy requirements first
WORKDIR /workspace

RUN uv python install 3.12

RUN uv venv

# Install all Python dependencies from requirements.txt
# --index-strategy unsafe-best-match resolves packaging dependency conflicts
RUN uv pip install runpod huggingface_hub hf_transfer llama-cpp-python --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu125"


ARG MODEL_ID=unsloth/Qwen3.5-0.8B-GGUF
ENV HF_HOME=/root/.cache/huggingface
ENV MODEL_PATH=/models/${MODEL_ID}

RUN mkdir -p /models && \
    uv run hf download ${MODEL_ID} \
    --local-dir ${MODEL_PATH} \
    --include "*mmproj-BF16*" \
    --include "*UD-IQ2_M*"

# Critical: Set offline mode AFTER the download to prevent network calls during runtime
ENV HF_HUB_OFFLINE=1

# Copy the serverless handler
COPY handler.py .

# Execute the handler script directly with unbuffered output
CMD ["uv", "run", "python", "-u", "handler.py"]