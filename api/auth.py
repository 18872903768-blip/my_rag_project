"""Service-to-service JWT authentication.

Set ``RAG_JWT_SECRET`` to enable verification; without it the service runs in
dev mode (no auth) and logs a warning.  The caller's role always comes from
the token's ``role`` claim so clients cannot escalate privileges via the
request body.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import jwt

logger = logging.getLogger(__name__)

DEFAULT_ROLE = "user"
VALID_ROLES = ("guest", "user", "staff")
DEFAULT_TOKEN_TTL_SECONDS = 24 * 3600


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity."""

    role: str
    subject: str | None = None
    dev_mode: bool = False


class AuthError(Exception):
    pass


def get_jwt_secret() -> str | None:
    return os.getenv("RAG_JWT_SECRET") or None


def mint_token(
    role: str, secret: str, *, subject: str | None = None, ttl: int = DEFAULT_TOKEN_TTL_SECONDS
) -> str:
    """Create a signed token — used by api.mint_token CLI and tests."""
    if role not in VALID_ROLES:
        raise ValueError(f"未知角色: {role!r}（可选 {VALID_ROLES}）")
    now = int(time.time())
    payload: dict[str, object] = {"role": role, "iat": now, "exp": now + ttl}
    if subject:
        payload["sub"] = subject
    return jwt.encode(payload, secret, algorithm="HS256")


def authenticate(authorization: str | None) -> Principal:
    """Verify the Bearer token and return the caller's role."""
    secret = get_jwt_secret()
    if not secret:
        logger.warning("RAG_JWT_SECRET 未设置，服务运行在无鉴权开发模式")
        return Principal(role=DEFAULT_ROLE, dev_mode=True)

    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError("缺少 Bearer Token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token 已过期") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Token 无效") from exc

    role = str(payload.get("role", DEFAULT_ROLE))
    if role not in VALID_ROLES:
        raise AuthError(f"Token 中的角色不合法: {role!r}")
    return Principal(role=role, subject=payload.get("sub"))
