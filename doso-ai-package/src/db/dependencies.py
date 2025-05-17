"""
Database session management and dependencies
"""

from collections.abc import AsyncGenerator
from typing import Callable

from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

from ..config import settings
from .redis import get_redis

# Create async engine with connection pooling
engine = create_async_engine(
    settings.DATABASE_URI,
    future=True,
    echo=settings.SQL_ECHO,
    poolclass=AsyncAdaptedQueuePool,
    pool_pre_ping=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_redis_dep() -> Callable[[], Redis]:
    """FastAPI dependency for getting Redis client"""
    return get_redis
