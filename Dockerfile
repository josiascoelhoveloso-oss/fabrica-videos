FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
ENV PORT=7860 PASTA_DADOS=/tmp/trabalhos ESCALA_VIDEO=0.667 PRESET_X264=ultrafast
EXPOSE 7860
CMD ["python", "server.py"]
