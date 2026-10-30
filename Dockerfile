FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data FASTEMBED_CACHE_PATH=/data/models TIKTOKEN_CACHE_DIR=/srv/tiktoken HOME=/tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the tokenizer into the image so token counting works offline.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || echo "tiktoken download failed; approximate counts will be used"
COPY app ./app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
# Runtime state (config, uploads, SQLite, Chroma, the embedding model cache) lives in /data.
# The entrypoint starts as root, fixes /data ownership, and runs the server as "app" (setpriv is in util-linux).
RUN useradd --system --no-create-home --user-group app && mkdir -p /data && chown app:app /data
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/api/health')"
# uvicorn trusts X-Forwarded-* only from FORWARDED_ALLOW_IPS (set it to your reverse proxy).
ENV FORWARDED_ALLOW_IPS=127.0.0.1,::1
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
