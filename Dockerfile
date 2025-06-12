# ------------------------------
# STEP 1: 빌드 스테이지
# ------------------------------
# 빌드 도구를 포함하는 devel 이미지를 사용합니다. (이전 runtime에서 변경됨)
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    PATH="/opt/venv/bin:$PATH"

# Python, venv, pip 및 빌드 필수 패키지 설치
# build-essential, cmake, libopenblas-dev, libblas-dev, liblapack-dev는
# 파이썬 과학 계산 라이브러리(예: faiss, torch) 빌드에 필요합니다.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python3-pip \
      python3-venv \
      build-essential \
      cmake \
      libopenblas-dev \
      libblas-dev \
      liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# 가상환경 생성
RUN python${PYTHON_VERSION} -m venv /opt/venv

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# 애플리케이션 의존성 설치
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 비권한 사용자 생성 (파일 복사 및 권한 설정 후 전환)
RUN adduser --no-create-home --disabled-login appuser


# ------------------------------
# STEP 2: 런타임 스테이지
# ------------------------------
# 더 작고 가벼운 런타임 이미지를 사용합니다.
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# 환경 변수 (가상환경 PATH 포함)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    PATH="/opt/venv/bin:$PATH"

# 최소 런타임 패키지 설치 (OpenMP 등, 빌드 단계에서 설치된 라이브러리 실행에 필요할 수 있음)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 빌드 스테이지에서 생성된 가상환경 복사
COPY --from=builder /opt/venv /opt/venv

# 소스 코드 복사 (USER 전환 전에 복사)
WORKDIR /app
COPY . .

# 비권한 사용자 전환
USER appuser

# Cloud Run 진입점 설정 (Gunicorn이 0.0.0.0:${PORT}에 바인딩되도록 수정)
# Gunicorn은 기본적으로 8000번 포트를 사용하려고 하지만, Cloud Run은 $PORT 환경 변수를 통해 포트를 지정합니다.
# 반드시 0.0.0.0:${PORT} 형식으로 바인딩해야 합니다.
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--workers", "1", "--threads", "8", "--timeout", "900", "main:app"]
