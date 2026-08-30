"""Mint a dev JWT for calling the RAG service.

Usage::

    python -m api.mint_token --role staff [--secret <secret>] [--ttl 86400]
"""

from __future__ import annotations

import argparse
import sys

from api.auth import VALID_ROLES, get_jwt_secret, mint_token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 RAG 服务的测试 JWT")
    parser.add_argument("--role", choices=VALID_ROLES, default="user")
    parser.add_argument("--subject", default=None, help="调用方标识（可选）")
    parser.add_argument("--ttl", type=int, default=24 * 3600)
    parser.add_argument("--secret", default=None, help="不传则取 RAG_JWT_SECRET 环境变量")
    args = parser.parse_args(argv)

    secret = args.secret or get_jwt_secret()
    if not secret:
        print("请通过 --secret 或 RAG_JWT_SECRET 提供签名密钥", file=sys.stderr)
        return 1

    token = mint_token(args.role, secret, subject=args.subject, ttl=args.ttl)
    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
