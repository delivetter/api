from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth_handler import decode_token


class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> dict[Any, Any]:
        try:
            credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        # TODO: To be removed when FastAPI merges https://github.com/tiangolo/fastapi/pull/2120
        except HTTPException as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e.detail)

        if (
            payload := decode_token(credentials.credentials)
        ) is None:  # Check if token is correct
            exception_details = (
                "The server could not verify that you are authorized to access the URL requested. "
                "You either supplied the wrong credentials (e.g. a bad password), or your browser "
                "doesn't understand how to supply the credentials required."
            )
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=exception_details)
        return payload
