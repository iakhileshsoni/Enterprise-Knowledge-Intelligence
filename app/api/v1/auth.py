from typing import Type, TypeVar

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.schemas.auth import TokenRequest, TokenResponse, UserCreate, UserResponse, RefreshRequest
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])

T = TypeVar("T", bound=BaseModel)


async def _parse_request_body(request: Request, model_cls: Type[T]) -> T:
    """Parse request body supporting both JSON and x-www-form-urlencoded."""
    content_type = request.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            raw = await request.json()
        else:
            form = await request.form()
            raw = dict(form)
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors())
    except Exception:
        raise RequestValidationError(
            [{"loc": ["body"], "msg": "Could not parse request body", "type": "value_error"}]
        )


_user_schema = UserCreate.model_json_schema()
_token_schema = TokenRequest.model_json_schema()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": _user_schema},
                "application/x-www-form-urlencoded": {"schema": _user_schema},
            },
            "required": True,
        }
    },
)
async def register(request: Request, db: Session = Depends(get_db)) -> UserResponse:
    """Register a new user account (accepts either JSON or x-www-form-urlencoded).

    Note: In production, restrict this endpoint or require admin approval.
    For the local development MVP it is open.
    """
    payload = await _parse_request_body(request, UserCreate)
    auth_service = AuthService(db)
    user = auth_service.register(
        username=payload.username,
        email=payload.email,
        password=payload.password,
    )
    return UserResponse.model_validate(user)


@router.post(
    "/token",
    response_model=TokenResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": _token_schema},
                "application/x-www-form-urlencoded": {"schema": _token_schema},
            },
            "required": True,
        }
    },
)
async def login(
    request: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Authenticate with username + password (accepts either JSON or x-www-form-urlencoded).

    Returns an access token (short-lived) and a refresh token (long-lived).
    """
    payload = await _parse_request_body(request, TokenRequest)
    auth_service = AuthService(db)
    user = auth_service.authenticate(payload.username, payload.password)
    return auth_service.create_token_pair(user)


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Exchange a valid refresh token for a new access token pair."""
    auth_service = AuthService(db)
    return auth_service.refresh_access_token(payload.refresh_token)


@router.get("/me", response_model=UserResponse)
def get_me(current_user=Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user's profile."""
    return UserResponse.model_validate(current_user)
