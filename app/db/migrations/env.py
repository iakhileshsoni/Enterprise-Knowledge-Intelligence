"""Alembic environment script.

This file is called by ``alembic upgrade``, ``alembic downgrade``, and other
Alembic commands.  It reads the database URL from the application settings
instead of from alembic.ini so that the same .env file governs everything.
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import Base and all models so Alembic autogenerate can detect them
from app.db.base import Base
import app.models  # noqa: F401 — registers all ORM models with Base.metadata

from app.core.config import get_settings

settings = get_settings()

# Alembic Config object, providing access to values within the .ini file
config = context.config

# Apply Python logging configuration from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Tell Alembic which metadata to compare against for autogenerate
target_metadata = Base.metadata

# Override the sqlalchemy.url from alembic.ini with the value from Settings
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    """Run migrations without an active database connection (offline mode)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with an active database connection (online mode)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # Required for SQLite ALTER TABLE support
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
