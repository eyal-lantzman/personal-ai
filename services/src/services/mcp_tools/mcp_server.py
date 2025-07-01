import os
import asyncio
from fastmcp import FastMCP, settings
from pathlib import Path
import concurrent
import threading
import uvicorn
import logging
import time
logger = logging.getLogger(__name__)

CODE_ROOT = os.path.abspath(Path(__file__).parent.parent.parent) + os.sep

class UviServerDecorator(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)

        # Remember the server instance associated with the test, assuming only one concurrent!
        setattr(UviServerDecorator, "value", self)

class MCPProjectServer:
    def __init__(self,
                root:str,
                enable_fs:bool=False,
                **mcp_kwargs):
        if not root:
            raise ValueError("root is required")
        
        if os.path.abspath(root).startswith(CODE_ROOT):
            raise ValueError("Dangerous root:" + root)
        
        self.mcp = FastMCP("local mcp", **mcp_kwargs)
        self.fs = None
        if "port" in mcp_kwargs:
            settings.port = mcp_kwargs["port"]
        if "host" in mcp_kwargs:
            settings.host = mcp_kwargs["host"]
        if "log_level" in mcp_kwargs:
            settings.log_level = mcp_kwargs["log_level"]
        
        if enable_fs:
            from services.mcp_tools.file_system import FileSystem
            self.fs = FileSystem(root)
            self.mcp.mount("fs", self.fs.mcp)

    def create_app(self, transport:str):
        self.app = self.mcp.http_app(transport=transport)
        self.generator = self.app.lifespan(self.app)
        self.generator_started = False
        return self.app

    async def start_session(self):
        if self.generator_started:
            return
        await self.generator.__aenter__()
        self.generator_started = True

    async def stop_session(self):
        if self.generator_started:
            await self.generator.__aexit__(None, None, None)
        self.generator_started = False
        self.generator = None
        self.app = None

    def start_uvicorn(self, transport:str):
        logging.info("Starting uvicorn")
        self.transport = transport
        self.pool = concurrent.futures.ThreadPoolExecutor()
        logging.info("Starting server from thread %d", threading.current_thread().native_id)
        self.server_future = asyncio.new_event_loop().run_in_executor(self.pool, self._serve)
        while(not hasattr(UviServerDecorator, "value")):
            time.sleep(1)
        return self.server_future
            
    def stop_uvicorn(self):
        logging.info("Stopping uvicorn")
        # Set the exit flag, ugly!
        if hasattr(UviServerDecorator, "value"):
            getattr(UviServerDecorator, "value").should_exit = True
        try:
            self.server_future.cancel()
            mcp_thread.join(2)
            self.pool.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logger.warning(e)

    def _serve(self):
        global mcp_thread
        mcp_thread = threading.current_thread()
        logging.info("Starting server on thread: %d", mcp_thread.native_id)
        try:
            # TODO: Nasty hack, there's no way to gracefully stop the server!
            from unittest.mock import  patch
            with patch("uvicorn.Server", new=UviServerDecorator) as mock:
                self.mcp.run(transport=self.transport)
        finally:
            logging.info("Stopped server")
