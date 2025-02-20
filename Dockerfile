FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    wget \
    git \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    libclang-14-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    openai \
    pyyaml \
    requests \
    typing \
    asyncio

WORKDIR /Log

RUN mkdir -p /Log/build

# Set environment variables
ENV PATH="/Log/build:${PATH}"
ENV PYTHONPATH="/Log:${PYTHONPATH}"

# Default command
CMD ["bash"]
