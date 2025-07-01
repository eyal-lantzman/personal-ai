import asyncio
import concurrent
import uuid
from pydantic import BaseModel, Field
from typing import Optional, Any
from collections.abc import Callable
import logging

logger = logging.getLogger(__name__)

class ActionResult(BaseModel):
    result: Optional[Any] = None
    last_error : Optional[Exception] = None
    cancelled: Optional[bool] = False
    completed: Optional[bool] = False
    succeeded: Optional[bool] = False

    class Config:
        arbitrary_types_allowed = True

class _Runnable(BaseModel):
    id: str
    func: Callable
    args: Optional[list] = Field(default_factory=list)
    kwargs : Optional[dict] = Field(default_factory=dict)
    result:ActionResult = Field(default_factory=ActionResult)
    task: Optional[asyncio.Task[ActionResult]] = None

    class Config:
        arbitrary_types_allowed = True
    

class ActionManager:
    def __init__(self, concurrency:int = 1, max_size:int = 0):
        """
        Initiatizes and Action Manager that is responsible to control the number of concurrent actions.

        Args:
            concurrency(int): The number of parallel or pseudo-parallel running actions. 
                                All other actions will be awaiting in queue and executed based on their priority.
            max_size(int):  If less than or equal to zero, the queue size is infinite.
                            If it is an integer greater than 0, then "await schedule()" will block when the
                            queue reaches max_size, until an action is finished running.
        """
        self.concurrency = concurrency
        self.tasks = dict[str, asyncio.Task[ActionResult]]()
        self.loop = asyncio.new_event_loop()
        self.loop.set_task_factory(asyncio.eager_task_factory)
        self.loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=concurrency))
        self.incoming = asyncio.queues.PriorityQueue[_Runnable](max_size)

        self.looping = False

    def schedule(self, func:Callable, *args, id:str=None, priority:int=10,  **kwargs) -> asyncio.Task[str]:
        """Schedule function call, and wait until scheduled.

        Args:
            func (function): The function to call
            id (str): The id of the function (meant for debugging and tracing purposes).
                        If non provided, will use method qualified name by default.
            priority(int): Optional priority, the lower the higher priority. Default priority is 10.
            args(list): Optional list of argument values.
            kwargs(dict): Optional dict of key-value arguments.

        Returns:
            asyncio.Task: Scheduling task, with and id of the execution task.
        """
        runnable = _Runnable(func=func, args=args, kwargs=kwargs, id=id or func.__qualname__ + str(uuid.uuid4()))
        runnable.result = ActionResult()

        async def _run():
            await self.incoming.put((priority, runnable))
            return runnable.id
        
        task = self.loop.create_task(_run(), name="enqueue")
        self.tasks[runnable.id] = None

        return task

    def _do_work(self, runnable:_Runnable) -> ActionResult:
        try:
            logger.debug("Running task %s", runnable.id)
            runnable.result.result = runnable.func(*runnable.args, **runnable.kwargs)
            runnable.result.succeeded = True
            logger.debug("Success in task %s", runnable.id)
        except asyncio.CancelledError as e:
            logger.debug("Cancelled task %s", runnable.id)
            runnable.result.last_error = e
            raise
        except Exception as e:
            logger.debug("Error in task %s", runnable.id)
            runnable.result.last_error = e
        finally:
            runnable.result.completed = True
            logger.debug("Completed task %s", runnable.id)
        return runnable.result

    async def _run(self):
        while self.looping:
            try:
                priority, work = await self.incoming.get()
                future = self.loop.run_in_executor(None, self._do_work, work)
                self.tasks[work.id] = future
                self.incoming.task_done()
            except asyncio.CancelledError as e:
                raise
            except Exception as e:
                logger.error(e)

    def start(self):
        self.looping = True
        self.background_run = self.loop.create_task(self._run(), name="background_run")

    def wait_for_empty_queue(self):
        while not self.incoming.empty():
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

    def wait_for_completion(self, awaitable:asyncio.Task | asyncio.Future):
        return self.loop.run_until_complete(awaitable)

    def wait_for_execution(self, id:str) -> ActionResult:
        if id not in self.tasks:
            raise ValueError("Id is not present in the queue: %s", id)
        
        while self.tasks[id] is None:
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

        return self.wait_for_completion(self.tasks[id])

    def stop(self, msg: Any|None = None) -> bool:
        if self.looping:
            self.looping = False
            asyncio.runners._cancel_all_tasks(self.loop)
            cancelled = self.background_run.cancel(msg)
            self.loop.stop()
            return cancelled
