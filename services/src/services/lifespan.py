import inspect
from typing import (
    Any, 
    AsyncContextManager, 
    AsyncIterator, 
    Callable, 
    ContextManager, 
    Generic,
    Iterable,
    Iterator, 
    Mapping, 
    Optional,
    Self, 
    TypeAlias,
    TypeVar, 
    Union, 
    cast
)
from types import MappingProxyType
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, suppress

TApp = TypeVar("TApp")


NoState: TypeAlias = None
State: TypeAlias = Mapping[str, Any]
AnyState = NoState | State

AnyContextManager: TypeAlias = Union[ContextManager[AnyState], AsyncContextManager[AnyState]]
AnyIterator: TypeAlias = Union[Iterator[AnyState], AsyncIterator[AnyState]]

RawLifespan: TypeAlias = Union[
    # (app, state) -> ...
    Callable[[TApp, State], Union[AnyContextManager, AnyIterator]],
    # (app) -> ...
    Callable[[TApp], Union[AnyContextManager, AnyIterator]],
    # () -> ...
    Callable[[], Union[AnyContextManager, AnyIterator]],
]
Lifespan: TypeAlias = Union[
    Callable[[TApp], AsyncContextManager[NoState]],
    Callable[[TApp], AsyncContextManager[State]],
]


def _maybe_call(call: Callable[..., AnyContextManager], app: Optional[TApp], state: State) -> AnyContextManager:
    with suppress(TypeError):
        return call(app, state)

    with suppress(TypeError):
        return call(app)

    return call()

def _convert_raw_lifespan_to_ctx(
    app: TApp,
    state: State,
    raw_lifespan: RawLifespan[TApp],
) -> Union[ContextManager[AnyState], AsyncContextManager[AnyState]]:
    if inspect.isasyncgenfunction(raw_lifespan):
        return _maybe_call(asynccontextmanager(raw_lifespan), app, state)
    if inspect.isgeneratorfunction(raw_lifespan):
        return _maybe_call(contextmanager(raw_lifespan), app, state)

    return _maybe_call(cast(Callable[..., AnyContextManager], raw_lifespan), app, state)


@asynccontextmanager
async def _run_raw_lifespan(
    raw_lifespan: RawLifespan[TApp],
    app: TApp,
    state: State,
) -> AsyncIterator[AnyState]:
    from fastapi.concurrency import contextmanager_in_threadpool
    ctx = _convert_raw_lifespan_to_ctx(app, state, raw_lifespan)

    actx: AsyncContextManager[AnyState]
    actx = contextmanager_in_threadpool(ctx) if isinstance(ctx, ContextManager) else ctx

    async with actx as res:
        yield res


class ManagedLifespan(Generic[TApp]):
    lifespans: list[RawLifespan[TApp]]

    def __init__(self, lifespans: Optional[Iterable[RawLifespan[TApp]]] = None, /) -> None:
        self.lifespans = [*(lifespans or [])]

    def add(self, lifespan: RawLifespan[TApp]) -> RawLifespan[TApp]:
        self.lifespans.append(lifespan)
        return lifespan

    def remove(self, lifespan: RawLifespan[TApp]) -> None:
        self.lifespans.remove(lifespan)

    def include(self, other: Self) -> None:
        self.lifespans.extend(other.lifespans)

    @asynccontextmanager
    async def __call__(self, app: TApp) -> AsyncIterator[AnyState]:
        async with AsyncExitStack() as astack:
            state: dict[str, Any] = {}
            state_proxy: State = MappingProxyType(state)

            for raw_lifespan in self.lifespans:
                sub_state = await astack.enter_async_context(_run_raw_lifespan(raw_lifespan, app, state_proxy))

                if sub_state:
                    state.update(sub_state)

            yield state
