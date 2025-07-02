import asyncio
import concurrent
import uuid
import functools
import time
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

@functools.total_ordering
class _Runnable(BaseModel):
    id: str
    func: Callable
    callback: Optional[Callable] = None
    timestamp: Optional[int] = Field(default_factory=time.monotonic_ns)
    args: Optional[list] = Field(default_factory=list)
    kwargs : Optional[dict] = Field(default_factory=dict)
    result:ActionResult = Field(default_factory=ActionResult)
    #task: Optional[asyncio.Task[ActionResult]] = None

    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    class Config:
        arbitrary_types_allowed = True
    

class ActionManager:
    def __init__(self, concurrency:int = 1, max_size:int = 0, force_new_loop:bool = False):
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
        if force_new_loop:
            self.loop = asyncio.new_event_loop()
        else:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()

        self.loop.set_task_factory(asyncio.eager_task_factory)
        self.loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=concurrency))
        self.incoming = asyncio.queues.PriorityQueue[_Runnable](max_size)
        self.outgoing = dict[str, asyncio.Task[ActionResult]]()
        self.cancelled = set[str]()
        self.looping = False

    def _scheduled(self, future:asyncio.Task[str], id:str):
        future.remove_done_callback(self._scheduled)
        if future.cancelled():
            self.outgoing.pop(id, None)

    def schedule(self, func:Callable, *args, id:str=None, priority:int=10, callback=None, **kwargs) -> asyncio.Task[str]:
        """Schedule function call, and doesn't wait until fully scheduled.
        To cancel the scheduling, use the returned task.
        To get the assigned id of the scheduled task, use `await_for_completion(task)` or `wait_for_completion(task)`.
        To get the final execution result, use `await_for_execution(id)` or `wait_for_execution(id)`.

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
        runnable_id = id or func.__qualname__ + str(uuid.uuid4())
        runnable = _Runnable(func=func, args=args, kwargs=kwargs, id=runnable_id, callback=callback)
        runnable.result = ActionResult()

        async def _enqueue():
            await self.incoming.put((priority, runnable))
            return runnable.id

        task = self.loop.create_task(_enqueue(), name="enqueue")
        task.add_done_callback(functools.partial(self._scheduled, id=runnable_id))
        self.outgoing[runnable.id] = None

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
            if runnable.callback is not None:
                self.loop.call_soon(runnable.callback, runnable.result)
        return runnable.result

    def _executed(self, future:asyncio.Future[ActionResult], id:str)-> object:
        future.remove_done_callback(self._executed)
        # If scheduled task reached this far, make sure it's in ends up in the outgoing
        if id not in self.outgoing:
            self.outgoing[id] = future

    async def _run(self):
        while self.looping:
            try:
                priority, work = await self.incoming.get()
                future = self.loop.run_in_executor(None, self._do_work, work)
                future.add_done_callback(functools.partial(self._executed, id=work.id))
                #work.task = future
                self.outgoing[work.id] = future
                self.incoming.task_done()
            except asyncio.CancelledError as e:
                raise

    def start(self):
        """Start the action manager.
        This will start the background task that will process the scheduled tasks."""
        self.looping = True
        self.background_run = self.loop.create_task(self._run(), name="background_run")

    def wait_for_empty_queue(self):
        """Wait for the incoming queue to be empty.
        This is useful to ensure that all scheduled tasks have been processed.
        """
        while not self.incoming.empty():
            self.loop.run_until_complete(asyncio.sleep(1))

    def wait_for_completion(self, awaitable:asyncio.Task | asyncio.Future) -> Any:
        """Wait for the completion of the given awaitable.
        Args: 
            awaitable (asyncio.Task | asyncio.Future): The awaitable to wait for.
        Returns: 
            Any: The result of the action or None if the awaitable was cancelled.
        """
        # Run the awaitable in the event loop and return the result
        if awaitable.cancelled():
            return None
        return self.loop.run_until_complete(awaitable)

    async def await_for_completion(self, awaitable:asyncio.Task | asyncio.Future) -> Any:
       """Wait for the completion of the given awaitable.
       Args:
            awaitable (asyncio.Task | asyncio.Future): The awaitable to wait for.
        Returns: 
            Any: The result of the action.
       """
       return await awaitable

    async def await_for_execution(self, id:str, keep_storing_result:bool=False) -> ActionResult:
        """Wait for the execution of the action with the given id.
        Args:
            id (str): The id of the action to wait for.
            keep_storing_result (bool): If True, the result will be kept in the outgoing dict.
                                        If False, the result will be removed from the outgoing dict.
        Returns:
            ActionResult: The result of the action, or None if the action was not found.
        """
        if id not in self.outgoing:
            return None
          
        missing = object()
        # None will indigate that we have a key but the value is None, 
        # and missing will indicate that the key is missing.
        # This way we can avoid KeyError due to race conditions when
        # called from multiple threads and popped the id in one of them.
        try:
            awaitable = self.outgoing.get(id, missing)
            while awaitable is None and awaitable is not missing:
                await asyncio.sleep(0)
                awaitable = self.outgoing.get(id, missing)
        except KeyError:
            # Could be a race condition between the first IF and the get(id,...)
            return None
        
        if awaitable is None or awaitable is missing:
            # Some other request popped it
            return None

        try:
            return await self.await_for_completion(awaitable)
        finally:
            if not keep_storing_result:
                self.outgoing.pop(id, None)

    def _get_awaitable(self, id:str) -> asyncio.Task | None:
        missing = object()
        try:
            # The future is set after the scheduling tasks completed 
            # i.e. scheduled for execution
            awaitable = self.outgoing.get(id, missing)
            while awaitable is None and awaitable is not missing:
                self.loop.run_until_complete(asyncio.sleep(1))
                awaitable = self.outgoing.get(id, missing)
        except KeyError:
            # Could be a race condition between the first IF and the get(id,...)
            return None
        
        if awaitable is None or awaitable is missing:
            return None
        
        return self.outgoing.get(id, None)
    
    @functools.lru_cache
    def wait_for_execution(self, id:str, keep_storing_result:bool=False) -> ActionResult:
        """Wait for the execution of the action with the given id.
        Args:
            id (str): The id of the action to wait for.
            keep_storing_result (bool): If True, the result will be kept in the outgoing dict.
                                        If False, the result will be removed from the outgoing dict.
        Returns:
            ActionResult: The result of the action, or None if the action was not found.
        """
        if id not in self.outgoing:
            return None
        
        awaitable = self._get_awaitable(id)
        
        if awaitable is None:
            return None
        
        try:
            return self.wait_for_completion(awaitable)
        finally:
            if not keep_storing_result:
                self.outgoing.pop(id, None)
    
    def cancel(self, id:str) -> bool:
        """Cancel the action with the given id.

        Args:
            id (str): The id of the action to cancel.

        Returns:
            bool: True if the action was cancelled, False otherwise.
        """
        if id not in self.outgoing:
            return False
        
        awaitable = self._get_awaitable(id)
        
        return awaitable.cancel()

    def cancel_all(self, msg: Any|None = None) -> bool:
        """Cancel all actions in the action manager.

        Args:
            msg (Any|None): Optional message to pass to the cancellation.
        
        Returns:
            bool: True if any action was cancelled, False otherwise.
        """
        cancelled = False
        for id in list(self.outgoing.keys()):
            cancelled |= self.cancel(id)
        
        return cancelled

    def stop(self, msg: Any|None = None) -> bool:
        """Stop the action manager and cancel all running tasks.
        Args:
            msg (Any|None): Optional message to pass to the cancellation.
        Returns:
            bool: True if the action manager was stopped, False if it was not running.
        """
        if self.looping:
            self.looping = False
            cancelled = self.background_run.cancel(msg)
            self.loop.call_soon_threadsafe(asyncio.runners._cancel_all_tasks, self.loop)
            return cancelled
