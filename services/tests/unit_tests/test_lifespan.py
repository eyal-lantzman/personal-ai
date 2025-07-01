from contextlib import asynccontextmanager, contextmanager

import pytest
from asgi_lifespan import LifespanManager as ASGILifespanManager
from fastapi import FastAPI

from services.lifespan import ManagedLifespan

STATE = {"foo": "bar"}
NO_STATE = None


async def _async_gen_no_state(_):
    yield


async def _async_gen_with_state(_):
    yield STATE


@asynccontextmanager
async def _async_ctx_no_state(_):
    yield


@asynccontextmanager
async def _async_ctx_with_state(_):
    yield STATE


def _gen_no_state(_):
    yield


def _gen_with_state(_):
    yield STATE


@contextmanager
def _ctx_no_state(_):
    yield


@contextmanager
def _ctx_with_state(_):
    yield STATE


async def _async_gen_no_state_no_app():
    yield


async def _async_gen_with_state_no_app():
    yield STATE


@asynccontextmanager
async def _async_ctx_no_state_no_app():
    yield


@asynccontextmanager
async def _async_ctx_with_state_no_app():
    yield STATE


def _gen_no_state_no_app():
    yield


def _gen_with_state_no_app():
    yield STATE


@contextmanager
def _ctx_no_state_no_app():
    yield


@contextmanager
def _ctx_with_state_no_app():
    yield STATE


async def get_state(manager: ManagedLifespan):
    app = FastAPI(lifespan=manager)

    async with ASGILifespanManager(app) as asgi_manager:
        return {**asgi_manager._state}


@pytest.mark.parametrize(
    ("raw_lifespan", "has_state"),
    [
        (_async_gen_no_state, False),
        (_async_gen_with_state, True),
        (_async_ctx_no_state, False),
        (_async_ctx_with_state, True),
        (_gen_no_state, False),
        (_gen_with_state, True),
        (_ctx_no_state, False),
        (_ctx_with_state, True),
        (_async_gen_no_state_no_app, False),
        (_async_gen_with_state_no_app, True),
        (_async_ctx_no_state_no_app, False),
        (_async_ctx_with_state_no_app, True),
        (_gen_no_state_no_app, False),
        (_gen_with_state_no_app, True),
        (_ctx_no_state_no_app, False),
        (_ctx_with_state_no_app, True),
    ],
)
@pytest.mark.asyncio
async def test_lifespan(raw_lifespan, has_state):
    under_test = ManagedLifespan([raw_lifespan])

    state = await get_state(under_test)
    assert state == (STATE if has_state else {})


@pytest.mark.asyncio
async def test_add():
    under_test = ManagedLifespan()

    called = False

    @under_test.add
    async def _lifespan(_):
        nonlocal called
        called = True
        yield

    assert not called

    await get_state(under_test)

    assert called


@pytest.mark.asyncio
async def test_state_inside_lifespan():
    under_test = ManagedLifespan()

    called = 0

    @under_test.add
    async def _lifespan_1(app, state):
        nonlocal called
        assert state == {}
        called += 1
        yield {"foo": "bar"}

    @under_test.add
    async def _lifespan_2(app, state):
        nonlocal called
        assert state == {"foo": "bar"}
        called += 1
        yield

    await get_state(under_test)

    assert called == 2


@pytest.mark.asyncio
async def test_remove():
    under_test = ManagedLifespan()

    called = False

    @under_test.add
    async def _lifespan(_):
        nonlocal called
        called = True
        yield

    assert not called

    under_test.remove(_lifespan)

    await get_state(under_test)

    assert not called


@pytest.mark.asyncio
async def test_include():
    calls = 0

    def _lifespan():
        nonlocal calls
        calls += 1
        yield

    parent = ManagedLifespan([_lifespan])
    child = ManagedLifespan([_lifespan])

    await get_state(parent)
    assert calls == 1

    parent.include(child)

    await get_state(parent)
    assert calls == 3