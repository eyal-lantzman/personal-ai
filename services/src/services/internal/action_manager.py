import asyncio
from pydantic import BaseModel, Field
from typing import Optional, Any

class ActionResult(BaseModel):
    result: Optional[Any] = None
    last_error : Optional[Exception] = None
    cancelled: Optional[bool] = False
    completed: Optional[bool] = False
    succeeded: Optional[bool] = False


class _Runnable(BaseModel):
    id: str
    func: function
    args: Optional[list] = Field(default_factory=list)
    kwargs : Optional[dict] = Field(default_factory=dict)
    # retry_on_exceptions: Optional[list[Exception]] = Field(default_factory=list)
    # max_retries: Optional[int] = 0
    task = Optional[asyncio.Task[ActionResult]] = None
    result: Optional[ActionResult] = None
    

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
        self.queue = asyncio.queues.PriorityQueue[_Runnable](max_size)
        self.loop = asyncio.new_event_loop()
        self.looping = False

    def schedule(self, func:function, id:str=None, priority:int=10, *args, **kwargs) -> asyncio.Task[ActionResult]:
        """Schedule function call, and wait until scheduled.

        Args:
            func (function): The function to call
            id (str): The id of the function (meant for debugging and tracing purposes).
                        If non provided, will use method qualified name by default.
            priority(int): Optional priority, the lower the higher priority. Default priority is 10.
            args(list): Optional list of argument values.
            kwargs(dict): Optional dict of key-value arguments.

        Returns:
            None
        """
        runnable = _Runnable(func=func, args=args, kwargs=kwargs, id=id or func.__qualname__)
        runnable.result = ActionResult()

        async def _run():
            await self.queue.put((priority, runnable))
            return runnable.result
        
        runnable.task = asyncio.create_task(_run)

        return runnable.task

    def _run_work(self, work:_Runnable) -> Any:
        try:
            result = work.func(work.args, work.kwargs)
            work.result.result = result
            work.result.succeeded = True
        except Exception as e:
            work.result.last_error = e
        finally:
            work.result.completed = True


    async def _run(self):
        while self.looping:
            work = await self.queue.get()
            self._run_work(work)

    def start(self):
        self.looping = True
        self.main_loop = asyncio.create_task(self._run)

    def stop(self) -> bool:
        self.looping = False
        return self.main_loop.cancel()