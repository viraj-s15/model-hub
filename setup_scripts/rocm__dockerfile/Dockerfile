ARG FROM_IMAGE_NAME=rocm/pytorch:latest
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  infiniband-diags \
  pciutils \
  numactl \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace/translation

COPY requirements.txt .
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
  && pip install --no-cache-dir -r requirements.txt

# Copy and build Transformer
COPY . .
RUN pip install -e .
RUN ./install_requirements.sh

