# STEP 1: 빌드 스테이지 - CUDA 개발 도구와 Python이 포함된 NVIDIA 베이스 이미지 사용
# L4 GPU는 최신 CUDA 버전에 최적화되어 있습니다.
# 여기서는 CUDA 12.3.1 (가장 최신 안정 버전 중 하나)과 Ubuntu 22.04를 예시로 사용합니다.
# 사용하려는 Faiss-GPU 버전과 호환되는 CUDA 버전을 선택해야 합니다.
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# 시스템 업데이트 및 필요한 패키지 설치
# Python, pip, build-essential (컴파일러), git, cmake (Faiss 빌드용),
# OpenBLAS/LAPACK (수치 계산 라이브러리, Faiss 의존성)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    cmake \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 가상 환경 생성 및 활성화
# 모든 Python 의존성을 이 가상 환경에 설치하여 격리성을 높입니다.
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 의존성 설치
# Faiss-GPU는 CUDA 런타임 및 개발 도구에 의존하므로, 이 빌드 스테이지에서 설치해야 합니다.
COPY requirements.txt .
# pip 설치 명령은 requirements.txt에 `faiss-gpu-cu12` (또는 `faiss-gpu-cu11`)이 있다고 가정합니다.
RUN pip install --no-cache-dir -r requirements.txt \
    # 특정 버전의 torch를 CUDA와 명시적으로 매칭시켜야 할 경우 여기에 추가
    # 예: pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
    # 이는 requirements.txt의 torch 버전을 덮어쓸 수 있으므로 주의
    && echo "Python packages installed successfully."

# 2. 보안 목적의 사용자 생성 (빌드 스테이지에서 미리 생성)
# 런타임 스테이지에서도 이 사용자를 사용할 것입니다.
RUN adduser --no-create-home --disabled-login appuser

# STEP 2: 런타임 스테이지 - CUDA 런타임과 필요한 최소한의 파일만 포함하여 이미지 크기 최적화
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# 환경 변수 설정 (빌드 스테이지와 동일하게 경로 설정)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV PATH="/opt/venv/bin:$PATH"

# 런타임에 필요한 최소한의 시스템 라이브러리 설치
# `libgomp1`은 OpenMP 런타임 라이브러리로, PyTorch/NumPy 등에서 병렬 처리에 사용될 수 있습니다.
# 다른 라이브러리 (예: BLAS/LAPACK 런타임)는 이미 CUDA 이미지에 포함되어 있거나,
# 파이썬 패키지(예: numpy)가 동적으로 링크할 수 있습니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 빌드 스테이지에서 생성된 파이썬 가상 환경 복사
COPY --from=builder /opt/venv /opt/venv

# Cloud SQL Unix socket 연결용 디렉토리 생성
RUN mkdir -p /cloudsql

# VOLUME 설정 (선택적) - Cloud Run에서 볼륨 마운트는 외부 파일 시스템이 아닌 임시 저장소 목적
VOLUME ["/cloudsql"]

# 파이썬 import 경로
ENV PYTHONPATH=/app

# 애플리케이션 소스 복사 (마지막에 복사하여 빌드 캐시 효율성 높임)
WORKDIR /app
COPY . .

# 비권한 사용자로 전환 (보안 강화)
USER appuser

# 애플리케이션 실행 (PORT는 Cloud Run에서 자동 설정)
# `--workers 1`: GPU를 단일 워커가 독점하도록 하여 경합을 줄이고 효율을 높입니다.
# `--threads 8`: Gunicorn 워커 내에서 사용할 스레드 수. CPU 바운드 작업에 유용할 수 있습니다.
# `--timeout 900`: 뉴스 처리 및 임베딩 계산에 충분한 시간 (필요에 따라 조정).
# `main:app`: Flask 애플리케이션 인스턴스가 `main.py` 파일의 `app` 변수에 있다고 가정.
CMD sh -c "gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 main:app"