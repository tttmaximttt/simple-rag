#!/usr/bin/env bash
# End-to-end tests against a fresh RAG Studio stack.
#
#   e2e/run.sh                 docker compose stack (app + pgvector + fake models), torn down afterwards
#   e2e/run.sh --local         no docker: uvicorn + embedded Chroma + fake models, on this machine
#   e2e/run.sh --keep          leave the stack running afterwards (for debugging)
#   e2e/run.sh -k mcp -x       anything else goes to pytest
#
# Real models instead of the fakes: E2E_EMBEDDINGS=openai|local and E2E_LLM=openai|anthropic,
# with OPENAI_API_KEY / ANTHROPIC_API_KEY set. Other settings: see e2e/conftest.py.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=docker
KEEP=${E2E_KEEP:-0}
ARGS=()
for a in "$@"; do
  case "$a" in
    --local) MODE=local ;;
    --keep) KEEP=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

WORK=$PWD/e2e/.work
rm -rf "$WORK/import" "$WORK/data"
mkdir -p "$WORK/import" "$WORK/data"
PORT=${E2E_PORT:-18080}
export E2E_IMPORT_DIR=$WORK/import
export E2E_ADMIN_PASSWORD=${E2E_ADMIN_PASSWORD:-e2e-temporary-pw}
export E2E_BASE_URL=http://localhost:$PORT

# Python env: the active virtualenv, or ./.venv (created with python3.12 if missing).
if [ -z "${VIRTUAL_ENV:-}" ]; then
  [ -d .venv ] || "${PYTHON:-python3.12}" -m venv .venv
  . .venv/bin/activate
fi
pip install -q -r e2e/requirements.txt
[ "$MODE" = local ] && pip install -q -r requirements.txt
# E2E_CHROMIUM: path to a Chromium binary to use instead of Playwright's own download.
[ -n "${E2E_CHROMIUM:-}" ] || python -m playwright install chromium

if [ "$MODE" = docker ]; then
  COMPOSE=(docker compose -p rag-e2e -f docker-compose.yml -f e2e/docker-compose.e2e.yml)
  [ "${E2E_VECTOR_STORE:-}" = qdrant ] && COMPOSE+=(--profile qdrant)
  cleanup() {
    status=$?
    [ $status -ne 0 ] && "${COMPOSE[@]}" logs --tail 80 app || true
    if [ "$KEEP" = 1 ]; then
      echo "Stack left running at $E2E_BASE_URL (admin password: ${E2E_PASSWORD:-e2e-admin-password-1})."
      echo "Stop it with: ${COMPOSE[*]} down -v"
    else
      "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
    fi
    exit $status
  }
  "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
  trap cleanup EXIT
  BUILD=(--build)
  [ "${E2E_BUILD:-1}" = 0 ] && BUILD=()
  PORT=$PORT "${COMPOSE[@]}" up -d --wait "${BUILD[@]}"
else
  FAKE_PORT=${E2E_FAKE_PORT:-19000}
  export E2E_FAKE_URL=http://127.0.0.1:$FAKE_PORT
  export E2E_VECTOR_STORE=${E2E_VECTOR_STORE:-chroma}
  (cd e2e && exec python fake_models.py --host 127.0.0.1 --port "$FAKE_PORT") > "$WORK/fake-models.log" 2>&1 &
  FAKE_PID=$!
  # ALLOW_PRIVATE_URLS: the URL import tests fetch PDFs from the fake model server on localhost.
  DATA_DIR=$WORK/data IMPORT_DIR=$WORK/import ADMIN_PASSWORD=$E2E_ADMIN_PASSWORD ALLOW_PRIVATE_URLS=1 \
    python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" > "$WORK/app.log" 2>&1 &
  APP_PID=$!
  cleanup() {
    status=$?
    [ $status -ne 0 ] && tail -n 80 "$WORK/app.log" || true
    if [ "$KEEP" = 1 ]; then
      echo "App left running at $E2E_BASE_URL (pid $APP_PID, fake models pid $FAKE_PID)."
    else
      kill "$APP_PID" "$FAKE_PID" 2>/dev/null || true
    fi
    exit $status
  }
  trap cleanup EXIT
  for _ in $(seq 120); do
    curl -fsS "$E2E_BASE_URL/api/health" >/dev/null 2>&1 && break
    kill -0 "$APP_PID" 2>/dev/null || { echo "The app exited during startup:"; exit 1; }
    sleep 0.5
  done
fi

python -m pytest e2e ${ARGS[@]+"${ARGS[@]}"}
