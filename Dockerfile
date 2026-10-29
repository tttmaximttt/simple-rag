FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data FASTEMBED_CACHE_PATH=/data/models TIKTOKEN_CACHE_DIR=/srv/tiktoken
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the tokenizer into the image so token counting works offline.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || echo "tiktoken download failed; approximate counts will be used"
COPY app ./app
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
