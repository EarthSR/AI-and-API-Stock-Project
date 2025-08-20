# Multi-stage Docker build for AI Stock Prediction System
FROM continuumio/miniconda3:latest AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    git \
    libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file first for better layer caching
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]

# Copy TA-Lib wheel and install it
COPY TA_Lib-0.4.28-cp312-cp312-win_amd64.whl .
RUN pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl && \
    rm TA_Lib-0.4.28-cp312-cp312-win_amd64.whl

# Production stage
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies for production
RUN apt-get update && apt-get install -y \
    libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from base stage
COPY --from=base /opt/conda/envs/pytorch /opt/conda/envs/pytorch

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/Preproces/retrain_logs \
             /app/Preproces/mini_retrain_state \
             /app/Preproces/backup_models \
             /app/Preproces/online_models \
             /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CONDA_DEFAULT_ENV=pytorch
ENV PATH=/opt/conda/envs/pytorch/bin:$PATH

# Copy and set up configuration
COPY Preproces/config.env /app/Preproces/config.env.template

# Create a startup script
RUN echo '#!/bin/bash\n\
# Activate conda environment\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate pytorch\n\
\n\
# Check if config.env exists, if not copy from template\n\
if [ ! -f /app/Preproces/config.env ]; then\n\
    cp /app/Preproces/config.env.template /app/Preproces/config.env\n\
    echo "Please update /app/Preproces/config.env with your database credentials"\n\
fi\n\
\n\
# Run the main application\n\
cd /app/Preproces\n\
exec python Autotrainmodel.py "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD conda run -n pytorch python -c "import tensorflow as tf; import torch; print('OK')" || exit 1

# Expose ports if needed for API
EXPOSE 5000 8000

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []

# Labels
LABEL maintainer="AI Stock Prediction System"
LABEL description="Multi-model stock prediction system with LSTM, GRU, and XGBoost ensemble"
LABEL version="1.0"