FROM python:3.12-slim
RUN mkdir -p /root/sdt
WORKDIR /root/sdt

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential cmake sudo curl nano git pkg-config \
    libhdf5-dev meson liblapack-dev libopenblas-dev
RUN pip install --break-system-packages --upgrade pip uv
COPY requirements.txt .

RUN uv pip install --system --break-system-packages -r requirements.txt
