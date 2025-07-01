import os
import tempfile
from typing import AsyncIterator
from anyio import get_cancelled_exc_class
from fastapi import FastAPI,  Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from services.registry import registry_router
from services.healthcheck import healthcheck_router
from services.chat import chat_router
from services.embeddings import embeddings_router
from services.models import models_router
from services.lm_studio_load_balancer import lm_studio_lb_router
from services.mcp_tools.mcp_server import MCPProjectServer
from services.lifespan import ManagedLifespan, State

lifespan = ManagedLifespan()

import logging

logger = logging.getLogger(__name__)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]

mcp_transport = os.getenv("MCP_TRANSPORT", "streamable-http")
mcp_debug = bool(os.getenv("MCP_DEBUG", "True"))

mcp_server = MCPProjectServer(
    root=os.getenv("MCP_ROOT", tempfile.gettempdir()),
    enable_fs=True,
)

@lifespan.add
async def setup_mcp(app: FastAPI) -> AsyncIterator[State]:
    try:
        await mcp_server.start_session()
        yield {"mcp": mcp_server.mcp }
        await mcp_server.stop_session()
    except get_cancelled_exc_class():
        raise

app = FastAPI(lifespan=lifespan)

# TODO: is not really production quaility, this needs to be pulled from some secret vault instead.
app.add_middleware(SessionMiddleware, secret_key="some-random-string")

@app.middleware("http")
async def reuse_session(request: Request, call_next):
    response = await call_next(request)
    session = request.cookies.get("session")
    if session:
        response.set_cookie(key="session", value=request.cookies.get("session"), httponly=True)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/mcp", mcp_server.create_app(transport=mcp_transport), "MCP Server")
app.include_router(router=healthcheck_router, prefix='/health', tags=["Health Check"])
app.include_router(router=chat_router, prefix='/chat', tags=["OpenAI API compatible"])
app.include_router(router=embeddings_router, prefix='/embeddings', tags=["OpenAI API compatible"])
app.include_router(router=registry_router, prefix='/registry', tags=["Model Registry"])
app.include_router(router=models_router, prefix='/models', tags=["OpenAI API compatible"])
app.include_router(router=lm_studio_lb_router, prefix='/lm_studio_lb', tags=["LM Studio LB Proxy"])

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logger.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
