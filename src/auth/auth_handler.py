import logging
import time
from typing import Any, cast

from jose import JWTError, jwt

from .. import __project_name__
from ..config import CONFIG


def decode_token(token: str):
    JWT_ALGORITHM = "HS256"
    JWT_SECRET = CONFIG.jwt.secret

    try:
        return jwt.decode(token, JWT_SECRET, algorithms=JWT_ALGORITHM)
    except JWTError as e:
        print(e)
        logging.debug("Bearer token couldn't be decoded")
        return None


def create_token(
    subject: str, secret: str | None = None, expiration: int | None = None
) -> str:
    JWT_ALGORITHM = "HS256"
    JWT_SECRET = secret or CONFIG.jwt.secret
    now = int(time.time())
    tomorrow = now + 24 * 3600
    payload = {
        "iss": __project_name__,
        "sub": subject,
        "iat": now,
        "exp": expiration or tomorrow,
    }

    return cast(str, jwt.encode(payload, JWT_SECRET, JWT_ALGORITHM))
