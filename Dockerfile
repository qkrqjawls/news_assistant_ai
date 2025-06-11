# ------------------------------
# STEP 1: 빌드 스테이지
# ------------------------------
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS builder

# 환경 변수
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    PATH="/opt/venv/bin:$PATH"

# Python, venv, pip 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python3-pip \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

# 가상환경 생성 및 활성화
RUN python${PYTHON_VERSION} -m venv /opt/venv

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 애플리케이션 의존성 설치
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 비권한 사용자 생성
RUN adduser --no-create-home --disabled-login appuser


# ------------------------------
# STEP 2: 런타임 스테이지
# ------------------------------
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# 환경 변수 (가상환경 포함)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    PATH="/opt/venv/bin:$PATH"

# 최소 런타임 패키지 설치 (OpenMP 등)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 빌드 스테이지의 venv 복사
COPY --from=builder /opt/venv /opt/venv

# 소스 복사
WORKDIR /app
COPY . .

# 비권한 사용자 전환
USER appuser

# Cloud Run 진입점
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "900", "main:app"]
