from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, UserAlreadyExistsError
from app.core.logging import get_logger
from app.models.user import User
from app.repositories.user_repo import UserRepository
from app.schemas.auth import TokenPayload, TokenResponse

logger = get_logger(__name__)
settings = get_settings()


def hash_password(password: str) -> str:
    """Hash password using native bcrypt (handles 72-byte max cleanly)."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify password against bcrypt hash in constant time."""
    try:
        plain_bytes = plain.encode("utf-8")[:72]
        hashed_bytes = hashed.encode("utf-8")
        return bcrypt.checkpw(plain_bytes, hashed_bytes)
    except Exception:
        return False


class AuthService:
    """Authentication and token management service."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._user_repo = UserRepository(db)

    def register(self, username: str, email: str, password: str, role: str = "user") -> User:
        """Create a new user.  Raises UserAlreadyExistsError if username/email taken."""
        if self._user_repo.username_exists(username):
            raise UserAlreadyExistsError(f"Username {username!r} is already taken.")
        if self._user_repo.email_exists(email):
            raise UserAlreadyExistsError(f"Email {email!r} is already registered.")

        hashed = hash_password(password)
        user = self._user_repo.create(
            username=username,
            email=email,
            hashed_password=hashed,
            role=role,
        )
        logger.info("New user registered: %s (role=%s)", username, role)
        return user

    def authenticate(self, username: str, password: str) -> User:
        """Verify credentials. Raises AuthenticationError if invalid."""
        user = self._user_repo.get_by_username(username)
        if user is None or not verify_password(password, user.hashed_password):
            # Same error message for both cases to prevent username enumeration
            raise AuthenticationError("Invalid username or password.")
        if not user.is_active:
            raise AuthenticationError("Account is inactive. Contact an administrator.")
        return user

    def create_token_pair(self, user: User) -> TokenResponse:
        """Issue an access token + refresh token pair for the given user."""
        access_token = _create_jwt(
            sub=user.id,
            role=user.role,
            token_type="access",
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
        )
        refresh_token = _create_jwt(
            sub=user.id,
            role=user.role,
            token_type="refresh",
            expires_delta=timedelta(days=settings.refresh_token_expire_days),
        )
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60,
        )

    def get_user_from_token(self, token: str) -> Optional[User]:
        """Decode a JWT and return the corresponding User, or None if invalid."""
        payload = _decode_jwt(token, expected_type="access")
        if payload is None:
            return None
        return self._user_repo.get_by_id(payload.sub)

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Issue a new access token from a valid refresh token."""
        payload = _decode_jwt(refresh_token, expected_type="refresh")
        if payload is None:
            raise AuthenticationError("Invalid or expired refresh token.")
        user = self._user_repo.get_by_id(payload.sub)
        if user is None or not user.is_active:
            raise AuthenticationError("User not found or inactive.")
        return self.create_token_pair(user)


def _create_jwt(
    sub: str,
    role: str,
    token_type: str,
    expires_delta: timedelta,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "role": role,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def _decode_jwt(token: str, expected_type: str) -> Optional[TokenPayload]:
    try:
        raw = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        payload = TokenPayload(**raw)
        if payload.type != expected_type:
            return None
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Invalid JWT: %s", exc)
        return None
