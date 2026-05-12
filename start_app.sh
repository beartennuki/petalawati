#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
PREFECT_BIN="$VENV_DIR/bin/prefect"
UVICORN_BIN="$VENV_DIR/bin/uvicorn"
LOG_DIR="$ROOT_DIR/logs"
RUN_DIR="$ROOT_DIR/.run"
PREFECT_PORT=4200
APP_PORT=8000
WORK_POOL="cnn-pool"
DEPLOYMENT_NAME="training-flow/cnn-deploy"
SERVER_PID_FILE="$RUN_DIR/prefect-server.pid"
WORKER_PID_FILE="$RUN_DIR/prefect-worker.pid"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_DIR"

cleanup() {
  local exit_code=$?

  cleanup_service "${WORKER_PID:-}" "${WORKER_PID_FILE:-}" "Prefect worker"
  cleanup_service "${SERVER_PID:-}" "${SERVER_PID_FILE:-}" "Prefect server"

  exit "$exit_code"
}

cleanup_service() {
  local pid="${1:-}"
  local pid_file="${2:-}"
  local label="${3:-service}"
  local i

  if [[ -z "$pid" ]]; then
    [[ -n "$pid_file" ]] && rm -f "$pid_file"
    return
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    [[ -n "$pid_file" ]] && rm -f "$pid_file"
    return
  fi

  log "Stopping $label (PID $pid)."
  kill "$pid" 2>/dev/null || true

  for ((i=1; i<=10; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      [[ -n "$pid_file" ]] && rm -f "$pid_file"
      return
    fi
    sleep 1
  done

  log "$label did not exit after 10s; forcing shutdown."
  kill -9 "$pid" 2>/dev/null || true
  [[ -n "$pid_file" ]] && rm -f "$pid_file"
}

log() {
  printf '[start_app] %s\n' "$1"
}

print_banner() {
  cat <<'EOF'
██████╗ ███████╗████████╗ █████╗ ██╗      █████╗ ██╗    ██╗ █████╗ ████████╗██╗
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║     ██╔══██╗██║    ██║██╔══██╗╚══██╔══╝██║
██████╔╝█████╗     ██║   ███████║██║     ███████║██║ █╗ ██║███████║   ██║   ██║
██╔═══╝ ██╔══╝     ██║   ██╔══██║██║     ██╔══██║██║███╗██║██╔══██║   ██║   ██║
██║     ███████╗   ██║   ██║  ██║███████╗██║  ██║╚███╔███╔╝██║  ██║   ██║   ██║
╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝
EOF
}

fail() {
  printf '[start_app] Error: %s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

port_in_use() {
  lsof -ti "tcp:$1" >/dev/null 2>&1
}

pids_for_port() {
  lsof -ti "tcp:$1" 2>/dev/null | sort -u
}

kill_processes_on_port() {
  local port="$1"
  local pids
  local pid
  local i

  pids="$(pids_for_port "$port")"
  [[ -n "$pids" ]] || return 0

  log "Port $port is in use by PID(s): $(tr '\n' ' ' <<<"$pids" | xargs). Attempting shutdown."

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
  done <<<"$pids"

  for ((i=1; i<=10; i++)); do
    if ! port_in_use "$port"; then
      log "Port $port is free after graceful shutdown."
      return 0
    fi
    sleep 1
  done

  log "Port $port is still busy after 10s; forcing shutdown."
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -9 "$pid" 2>/dev/null || true
  done <<<"$pids"

  for ((i=1; i<=5; i++)); do
    if ! port_in_use "$port"; then
      log "Port $port is free after forced shutdown."
      return 0
    fi
    sleep 1
  done

  return 1
}

wait_for_port() {
  local port="$1"
  local label="$2"
  local attempts="${3:-60}"
  local i

  for ((i=1; i<=attempts; i++)); do
    if port_in_use "$port"; then
      log "$label is reachable on port $port."
      return 0
    fi
    sleep 1
  done

  fail "$label did not become reachable on port $port."
}

requirements_satisfied() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from importlib.metadata import packages_distributions, PackageNotFoundError
import re, sys

dist_names = {d.lower() for dists in packages_distributions().values() for d in dists}

requirements = Path("requirements.txt")
for raw_line in requirements.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    pkg = re.split(r"[>=<!;\[\s]", line)[0].strip().lower().replace("-", "_")
    if not pkg:
        continue
    if pkg not in dist_names:
        sys.exit(1)

sys.exit(0)
PY
}

prefect_api_ready() {
  "$PYTHON_BIN" - <<'PY'
import urllib.request, sys
try:
    with urllib.request.urlopen("http://127.0.0.1:4200/api/health", timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

ensure_venv() {
  [[ -x "$PYTHON_BIN" ]] || fail "Virtual environment not found at .venv/. Create it first."
  [[ -x "$PIP_BIN" ]] || fail "pip not found in .venv/bin."
  [[ -x "$PREFECT_BIN" ]] || fail "prefect not found in .venv/bin."
  [[ -x "$UVICORN_BIN" ]] || fail "uvicorn not found in .venv/bin."
}

ensure_dependencies() {
  if requirements_satisfied; then
    log "requirements.txt is already satisfied. Skipping pip install."
    return
  fi

  log "Installing missing Python dependencies from requirements.txt."
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

wait_for_prefect_api() {
  local attempts=60
  local i
  for ((i=1; i<=attempts; i++)); do
    if prefect_api_ready; then
      log "Prefect API is ready."
      return 0
    fi
    sleep 1
  done
  fail "Prefect API did not respond within ${attempts}s. Check $LOG_DIR/prefect-server.log."
}

start_prefect_server() {
  if prefect_api_ready; then
    log "Prefect server is already running and API is healthy."
    return
  fi

  if port_in_use "$PREFECT_PORT"; then
    if ! kill_processes_on_port "$PREFECT_PORT"; then
      fail "Port $PREFECT_PORT is in use but the Prefect API is not responding. Kill the process holding port $PREFECT_PORT and retry."
    fi
  fi

  log "Starting Prefect server."
  "$PREFECT_BIN" server start >"$LOG_DIR/prefect-server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" >"$SERVER_PID_FILE"
  wait_for_prefect_api
}

ensure_work_pool() {
  if "$PREFECT_BIN" work-pool inspect "$WORK_POOL" >/dev/null 2>&1; then
    log "Prefect work pool '$WORK_POOL' already exists."
    return
  fi

  log "Creating Prefect work pool '$WORK_POOL'."
  "$PREFECT_BIN" work-pool create "$WORK_POOL" --type process >/dev/null
}

ensure_deployment() {
  log "Deploying training flow from prefect.yaml."
  (
    cd "$ROOT_DIR"
    PYTHONPATH=. "$PREFECT_BIN" deploy --prefect-file prefect.yaml --all >/dev/null
  )
}

start_worker() {
  if pgrep -f "prefect worker start --pool $WORK_POOL" >/dev/null 2>&1; then
    log "A Prefect worker for pool '$WORK_POOL' is already running. Skipping worker start."
    return
  fi

  log "Starting Prefect worker for pool '$WORK_POOL'."
  PETALAWATI_ROOT="$ROOT_DIR" PYTHONPATH="$ROOT_DIR" \
    "$PREFECT_BIN" worker start --pool "$WORK_POOL" >"$LOG_DIR/prefect-worker.log" 2>&1 &
  WORKER_PID=$!
  printf '%s\n' "$WORKER_PID" >"$WORKER_PID_FILE"
}

start_web_app() {
  if port_in_use "$APP_PORT"; then
    fail "Port $APP_PORT is already in use. Stop the existing app or change the port."
  fi

  print_banner
  log "Starting FastAPI app on http://127.0.0.1:$APP_PORT"
  log "FastAPI URL: http://127.0.0.1:$APP_PORT"
  log "Prefect URL: http://127.0.0.1:$PREFECT_PORT"
  log "Logs: $LOG_DIR/prefect-server.log, $LOG_DIR/prefect-worker.log"
  trap cleanup EXIT INT TERM

  (
    cd "$ROOT_DIR"
    PYTHONPATH=. exec "$UVICORN_BIN" app.main:app --reload --port "$APP_PORT"
  )
}

main() {
  ensure_venv
  ensure_dependencies
  start_prefect_server
  ensure_work_pool
  ensure_deployment
  start_worker
  start_web_app
}

main "$@"
