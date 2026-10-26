from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    """Request body for user registration."""
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    """Safe user representation — never includes the hashed password."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime


class TokenRequest(BaseModel):
    """OAuth2 password grant form (used via form data in the /token endpoint)."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token pair returned on successful login."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # access token lifetime in seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPayload(BaseModel):
    """Decoded JWT payload structure."""
    sub: str          # user_id
    role: str
    type: str         # "access" | "refresh"
    exp: Optional[int] = None
