from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationError, PermissionDeniedError
from app.db.session import get_db
from app.models.user import User
from app.repositories.user_repo import UserRepository
from app.services.auth_service import AuthService

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: decode JWT, verify user exists and is active.

    Raises:
        AuthenticationError (401): Missing, invalid, or expired token.
    """
    if credentials is None:
        raise AuthenticationError("Authorization header is missing.")

    auth_service = AuthService(db)
    user = auth_service.get_user_from_token(credentials.credentials)
    if user is None:
        raise AuthenticationError("Invalid or expired access token.")
    if not user.is_active:
        raise AuthenticationError("User account is inactive.")
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency: verify the current user has the admin role.

    Raises:
        PermissionDeniedError (403): User is not an admin.
    """
    if current_user.role != "admin":
        raise PermissionDeniedError("This operation requires administrator privileges.")
    return current_user
