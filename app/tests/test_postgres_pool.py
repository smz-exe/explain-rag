"""Tests for connection pool lifecycle in the PostgreSQL adapters.

Both adapters lazily create their pool with a bare `if self._pool is None`
check. That await point is a yield: concurrent first-requests can each see None
and each create a pool, so all but the last are leaked — holding open
connections nobody will ever close. These tests pin single-pool creation and
the presence of timeouts.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.adapters.outbound.postgres_query_storage import PostgresQueryStorage
from src.adapters.outbound.postgres_vector_store import PostgresVectorStore

ADAPTERS = [PostgresVectorStore, PostgresQueryStorage]


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
@pytest.mark.asyncio
async def test_concurrent_first_use_creates_exactly_one_pool(adapter_cls):
    """Twenty simultaneous first requests must share one pool, not create twenty."""
    adapter = adapter_cls("postgresql://user:pass@localhost:5432/db")
    created = 0

    async def fake_create_pool(*args, **kwargs):
        nonlocal created
        created += 1
        # Yield control the way the real driver does while connecting, so a
        # check-then-act race actually manifests.
        await asyncio.sleep(0)
        return AsyncMock(name=f"pool-{created}")

    with patch("asyncpg.create_pool", side_effect=fake_create_pool):
        pools = await asyncio.gather(*(adapter._get_pool() for _ in range(20)))

    assert created == 1, f"created {created} pools; all but one would be leaked"
    assert all(pool is pools[0] for pool in pools)


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
@pytest.mark.asyncio
async def test_pool_is_created_with_a_command_timeout(adapter_cls):
    """Without a timeout a stalled query hangs the request until the client gives up."""
    adapter = adapter_cls("postgresql://user:pass@localhost:5432/db")

    with patch("asyncpg.create_pool", new=AsyncMock()) as create_pool:
        await adapter._get_pool()

    kwargs = create_pool.await_args.kwargs
    assert kwargs.get("command_timeout"), "pool must set command_timeout"
    assert kwargs.get("timeout"), "pool must bound how long acquiring a connection waits"


@pytest.mark.parametrize("adapter_cls", ADAPTERS)
@pytest.mark.asyncio
async def test_close_allows_a_later_pool_to_be_created(adapter_cls):
    """close() must reset the cached pool so the adapter stays reusable."""
    adapter = adapter_cls("postgresql://user:pass@localhost:5432/db")

    with patch("asyncpg.create_pool", new=AsyncMock()) as create_pool:
        await adapter._get_pool()
        await adapter.close()
        await adapter._get_pool()

    assert create_pool.await_count == 2
