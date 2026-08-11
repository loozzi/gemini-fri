# ─── Stage 1: build the virtualenv ────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is cached until
# pyproject.toml or uv.lock actually changes. No BuildKit cache mount here, so
# the image also builds with the classic builder (no buildx required).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev


# ─── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm

# uvloop and httptools land in the venv, so no compiler or uv is needed here.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app main.py ./
COPY --chown=app:app routers/ ./routers/
COPY --chown=app:app sdk/ ./sdk/

USER app

EXPOSE 8000

# Uses the venv's python rather than curl, which slim does not ship.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
