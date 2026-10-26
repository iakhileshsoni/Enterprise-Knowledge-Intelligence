from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models.

    All models import from this module so that ``alembic`` autogenerate and
    ``Base.metadata.create_all()`` can discover them from a single location.
    """
    pass
