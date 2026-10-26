from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
    """Data-access layer for User records.

    Contains no business logic — only database queries.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        username: str,
        email: str,
        hashed_password: str,
        role: str = "user",
    ) -> User:
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            role=role,
        )
        self._db.add(user)
        self._db.commit()
        self._db.refresh(user)
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self._db.get(User, user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        stmt = select(User).where(User.username == username)
        return self._db.execute(stmt).scalar_one_or_none()

    def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email)
        return self._db.execute(stmt).scalar_one_or_none()

    def username_exists(self, username: str) -> bool:
        return self.get_by_username(username) is not None

    def email_exists(self, email: str) -> bool:
        return self.get_by_email(email) is not None

    def deactivate(self, user_id: str) -> None:
        user = self.get_by_id(user_id)
        if user:
            user.is_active = False
            self._db.commit()
