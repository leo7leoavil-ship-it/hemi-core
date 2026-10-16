"""Security and API Authentication Module.

Implements token verification against API_AUTH_TOKEN via HTTP Bearer or X-API-Key header.
The check is fail-closed: if no token is configured, every request is rejected.
"""

import hmac
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from app.core.config import get_settings

security_bearer = HTTPBearer(auto_error=False)
security_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def is_valid_token(candidate: Optional[str]) -> bool:
    """Constant-time comparison of a candidate token against the configured one."""
    expected = get_settings().API_AUTH_TOKEN
    if not expected or not candidate:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


async def verify_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    api_key_header: Optional[str] = Security(security_api_key),
) -> str:
    """Validate incoming request against configured API_AUTH_TOKEN.
    
    Accepts either:
    1. Authorization: Bearer <API_AUTH_TOKEN>
    2. X-API-Key: <API_AUTH_TOKEN>
    """
    # Check Bearer token
    if credentials and is_valid_token(credentials.credentials):
        return credentials.credentials

    # Check X-API-Key header
    if is_valid_token(api_key_header):
        return api_key_header  # type: ignore[return-value]

    # If neither matched
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Invalid or missing API authorization token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_websocket_token(
    authorization: Optional[str], api_key: Optional[str], query_token: Optional[str]
) -> bool:
    """Validate a WebSocket handshake.

    Browsers cannot set custom headers on WebSockets, so the token may also be
    supplied via the ``token`` query parameter.
    """
    if authorization and authorization.lower().startswith("bearer "):
        if is_valid_token(authorization[7:].strip()):
            return True
    return is_valid_token(api_key) or is_valid_token(query_token)
