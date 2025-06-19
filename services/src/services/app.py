from fastapi import FastAPI,  Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.registry import registry_router
from services.healthcheck import healthcheck_router
from services.chat import chat_router
from services.embeddings import embeddings_router
from services.models import models_router

import logging

logger = logging.getLogger(__name__)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=healthcheck_router, prefix='/health')
app.include_router(router=chat_router, prefix='/chat')
app.include_router(router=embeddings_router, prefix='/embeddings')
app.include_router(router=registry_router, prefix='/registry')
app.include_router(router=models_router, prefix='/models')


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logger.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)