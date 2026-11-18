# Core Infrastructure & Configuration

## Overview
Provides foundational runtime configuration through Pydantic Settings, cryptographic token-based security verification, and structured application-wide logging.

## Interface
- [config.py:Settings]: Pydantic `BaseSettings` loading environment variables from `.env` and environment with thesis-aligned fallbacks.
- [config.py:get_settings]: Cached singleton function returning the active `Settings` instance.
- [security.py:is_valid_token]: Constant-time string comparison (`hmac.compare_digest`) against configured `API_AUTH_TOKEN`.
- [security.py:verify_api_token]: FastAPI dependency validating Bearer credentials or `X-API-Key` headers; raises HTTP 401 on failure.
- [security.py:verify_websocket_token]: Validates WebSocket connection tokens via Authorization header, `X-API-Key`, or `token` query parameter.
- [logging.py:setup_logging]: Configures structured root logging with level and format.
- [logging.py:logger]: Centralized application logger instance.

## Invariants
- [security.py:is_valid_token]: Fail-closed security policy: if `API_AUTH_TOKEN` is unset or empty, all incoming requests fail authentication.
- [config.py:Settings.MAX_AUDIO_BYTES]: Minimum accepted audio bound is 1024 bytes (default 25 MB).
- [config.py:Settings.ALLOWED_ORIGINS]: Supports comma-separated strings or JSON arrays; default `["*"]`.

## Side Effects
- [logging.py:setup_logging]: Attaches a `StreamHandler` emitting formatted logs to `sys.stdout`.
- [config.py:Settings]: Reads configuration files (`.env`) and system environment variables.

## Verification
- Run: `pytest`
