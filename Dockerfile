# FlowSight: dashboard + API in one CPU image (no GPU needed for the demo).
#
#   docker build -t flowsight .
#   docker run --rm -p 8501:8501 flowsight                                   # dashboard  -> http://localhost:8501
#   docker run --rm -p 8000:8000 flowsight uvicorn flowsight.api:app --host 0.0.0.0 --port 8000   # API -> /docs
#   docker run --rm flowsight python -m flowsight.pipeline --config configs/demo.yaml --live       # needs data/MOT17-04-raw.webm
#
# The image ships the precomputed outputs/ so the dashboard works immediately. To regenerate, run
# data/download_data.py then the pipeline / simulator inside the container (see README).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch first (the default wheel drags in ~2 GB of CUDA libraries), then everything else
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE "^(torch|torchvision)==" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

COPY . .
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
