import pytest
from unittest.mock import patch
import time
import asyncio
from services.internal.action_manager import ActionManager, ActionResult
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def wait_for_completion_mock():
    patcher = patch("services.internal.action_manager.ActionManager.wait_for_completion")
    mock = patcher.start()
    mock.return_value = ActionResult()
    yield mock
    patcher.stop()

@pytest.fixture
def under_test():
    test_subject = ActionManager(1)
    yield test_subject
    test_subject.stop()

def func_with_return_value():
    return 42

def func_with_arg(arg):
    return arg

def func_with_args(*args):
    return list(args)

def func_with_kwargs(**kwargs):
    return dict(kwargs)

def test_schedule_when_stopped(under_test:ActionManager):
    # Arrnage
    x = 0
    def inc():
        nonlocal x
        x+=1 

    # Act
    task = under_test.schedule(func=inc)

    # Assert
    assert not task.cancelled()
    assert not task.done()
    assert x == 0
    
    task.cancel()
    assert task.cancelling() > 0
    task.uncancel()
    assert task.cancelling() == 0

def test_schedule_with_no_return_value(under_test:ActionManager):
    # Arrnage
    x = 0
    def inc():
        nonlocal x
        x += 1
    under_test.start()

    # Act
    task = under_test.schedule(func=inc)
    id = under_test.wait_for_completion(task)
    outcome = under_test.wait_for_execution(id)

    # Assert
    assert not task.cancelled()
    assert task.done()
    assert x == 1
    assert outcome.result is None
    assert not outcome.cancelled
    assert outcome.completed
    assert outcome.last_error is None


def test_schedule_with_return_value(under_test:ActionManager):
    # Arrnage
    under_test.start()

    # Act
    task = under_test.schedule(func_with_return_value)
    id = under_test.wait_for_completion(task)
    outcome = under_test.wait_for_execution(id)

    # Assert
    assert not task.cancelled()
    assert task.done()
    assert outcome.result == 42
    assert not outcome.cancelled
    assert outcome.completed
    assert outcome.last_error is None

def test_schedule_with_arg(under_test:ActionManager):
    # Arrnage
    under_test.start()

    # Act
    task = under_test.schedule(func_with_arg, 5)
    id = under_test.wait_for_completion(task)
    outcome = under_test.wait_for_execution(id)

    # Assert
    assert not task.cancelled()
    assert task.done()
    assert outcome.result == 5
    assert not outcome.cancelled
    assert outcome.completed
    assert outcome.last_error is None


def test_schedule_with_args(under_test:ActionManager):
    # Arrnage
    under_test.start()

    # Act
    task = under_test.schedule(func_with_args, 5, 6)
    id = under_test.wait_for_completion(task)
    outcome = under_test.wait_for_execution(id)

    # Assert
    assert not task.cancelled()
    assert task.done()
    assert outcome.result == [5, 6]
    assert not outcome.cancelled
    assert outcome.completed
    assert outcome.last_error is None

def test_schedule_with_kwargs(under_test:ActionManager):
    # Arrnage
    under_test.start()

    # Act
    task = under_test.schedule(func_with_kwargs, key="value")
    id = under_test.wait_for_completion(task)
    outcome = under_test.wait_for_execution(id)

    # Assert
    assert not task.cancelled()
    assert task.done()
    assert outcome.result == {"key": "value"}
    assert not outcome.cancelled
    assert outcome.completed
    assert outcome.last_error is None

def test_schedule_with_exception(under_test:ActionManager):
    # Arrnage
    def throw_exception():
        raise Exception("boom")
    
    under_test.schedule(func=throw_exception, id="test1")

    # Act
    under_test.start()
    under_test.wait_for_empty_queue()

    outcome = under_test.wait_for_execution(id="test1")

    # Assert
    assert not outcome.succeeded
    assert outcome.completed
    assert not outcome.cancelled
    assert str(outcome.last_error) == "boom"

def test_schedule_with_priorities(under_test:ActionManager):
    # Arrnage
    counter = 0
    def increment_and_return():
        nonlocal counter
        counter += 1
        return counter
    
    under_test.schedule(func=increment_and_return, id="test1", priority=3)
    under_test.schedule(func=increment_and_return, id="test2", priority=2)
    under_test.schedule(func=increment_and_return, id="test3", priority=1)

    # Act
    under_test.start()
    under_test.wait_for_empty_queue()

    outcome1 = under_test.wait_for_execution(id="test1")
    outcome2 = under_test.wait_for_execution(id="test2")
    outcome3 = under_test.wait_for_execution(id="test3")

    # Assert
    assert outcome1.result == 3
    assert outcome2.result == 2
    assert outcome3.result == 1

def test_schedule_with_provided_id(under_test:ActionManager):
    # Arrnage
    under_test.start()

    # Act
    under_test.schedule(func_with_return_value, id="test1")
    under_test.schedule(func_with_return_value, id="test2")
    under_test.wait_for_empty_queue()
    outcome1 = under_test.wait_for_execution(id="test1", keep_storing_result=True)
    outcome2 = under_test.wait_for_execution(id="test1", keep_storing_result=False)
    outcome3 = under_test.wait_for_execution(id="test1")
    outcome4 = under_test.wait_for_execution(id="test2")

    # Assert
    assert outcome1 is outcome2
    assert outcome1.result == 42
    assert outcome3 is None
    assert outcome1 is not outcome4
    assert outcome4.result == 42


def test_schedule_with_callback(under_test:ActionManager):
    # Arrnage
    under_test.start()
    result = None
    def callback(action_result:ActionResult):
        nonlocal result
        result = action_result

    # Act
    under_test.schedule(func_with_return_value, id="test1", callback=callback)
    under_test.wait_for_empty_queue()
    outcome = under_test.wait_for_execution(id="test1")

    # Assert
    assert outcome is result
    assert result.result == 42

def test_schedule_with_provided_id_and_cached_retrieval(under_test:ActionManager, wait_for_completion_mock):
    # Arrnage
    under_test.start()

    # Act
    task = under_test.schedule(func_with_return_value, id="test1")
    under_test.wait_for_empty_queue()
    outcome1 = under_test.wait_for_execution(id="test1")
    outcome2 = under_test.wait_for_execution(id="test1")

    # Assert
    wait_for_completion_mock.assert_called_once()
    assert outcome1 is outcome2

@pytest.mark.parametrize("threads", [2,3,5,10])
def test_schedule_with_concurrency(threads):
    # Arrnage
    under_test = ActionManager(threads)
    def my_func():
        time.sleep(2)
    under_test.start()
    tasks = dict()
    outcomes = dict()
    start = time.time()
    # Two in parallel is too time sensitive to squeeze 2 runs under 4 seconds.
    concurrency = threads + 1 if threads > 2 else threads
    try:
        # Act
        for i in range(concurrency):
            task = under_test.schedule(my_func)
            tasks[task] = under_test.wait_for_completion(task)

        for scheduled, id in tasks.items():
            assert not scheduled.cancelled()
            assert scheduled.done()
            outcomes[id] = under_test.wait_for_execution(id)

        # Assert
        end = time.time()
        assert end - start < threads * 2
        for id, outcome in outcomes.items():
            assert outcome.result == None, id
            assert not outcome.cancelled, id
            assert outcome.completed, id
            assert outcome.last_error is None, id

    finally:
        under_test.stop()

@pytest.mark.asyncio
@pytest.mark.parametrize("loop", range(10))
async def test_schedule_with_concurrency_multiple_await_the_same_id(loop):
    # Arrnage
    threads = 10
    under_test = ActionManager(10)
    def my_func():
        time.sleep(1)
    under_test.start()
    tasks = dict()
    try:
        # Act
        for i in range(threads):
            task = under_test.schedule(my_func)
            tasks[task] = await under_test.await_for_completion(task)

        for scheduled, id in tasks.items():
            assert not scheduled.cancelled()
            assert scheduled.done()
            futures = await asyncio.gather(
                under_test.await_for_execution(id, keep_storing_result=True),
                under_test.await_for_execution(id, keep_storing_result=True),
            )
            
            for future in futures:
                assert future.result == None, id
                assert not future.cancelled, id
                assert future.completed, id
                assert future.last_error is None, id

            future = await under_test.await_for_execution(id)
            assert future.result == None, id
            assert not future.cancelled, id
            assert future.completed, id
            assert future.last_error is None, id

            future = await under_test.await_for_execution(id)
            assert future is None

        assert len(under_test.outgoing) == 0
    finally:
        under_test.stop()

@pytest.mark.parametrize("loop", range(10))
def test_schedule_and_cancel_scheduled(under_test:ActionManager, loop):
    # Arrnage
    counter = 0
    def inc():
        nonlocal counter
        counter += 1

    task = under_test.schedule(func=inc, id="task1", priority=1)

    # Act
    task.cancel()
    under_test.start()
    under_test.wait_for_empty_queue()
    outcome = under_test.wait_for_execution(id="task1")

    # Assert
    assert outcome is None
    assert counter == 0

@pytest.mark.parametrize("loop", range(10))
def test_schedule_and_cancel_execution(under_test:ActionManager, loop):
    # Arrnage
    counter = 0
    def inc():
        nonlocal counter
        time.sleep(2) # Simulate a long-running task
        counter += 1
        return counter
    
    under_test.schedule(func=func_with_return_value, id="task1", priority=1)
    under_test.schedule(func=func_with_return_value, id="task2", priority=2)
    under_test.schedule(func=inc, id="task3", priority=3)
    under_test.start()
    under_test.wait_for_empty_queue()
    assert under_test.cancel("task3")
    
    # Act
    outcome1 = under_test.wait_for_execution(id="task1")
    outcome3 = under_test.wait_for_execution(id="task2")
    outcome2 = under_test.wait_for_execution(id="task3")

    # Assert
    assert outcome1.result == 42
    assert not outcome1.cancelled
    assert outcome1.completed
    assert outcome1.last_error is None

    assert outcome2 is None
    assert counter == 0

    assert outcome3.result == 42
    assert not outcome3.cancelled
    assert outcome3.completed
    assert outcome3.last_error is None


@pytest.mark.parametrize("loop", range(10))
def test_schedule_and_cancel_all_execution(under_test:ActionManager, loop):
    # Arrnage
    counter = 0
    def task1():
        nonlocal counter
        time.sleep(2) # Simulate a long-running task
        counter += 1
        return counter

    def task2():
        nonlocal counter
        time.sleep(2) # Simulate a long-running task
        counter += 1
        return counter
    
    under_test.schedule(func=task1, id="task1", priority=1)
    under_test.schedule(func=task2, id="task2", priority=3)
    under_test.start()
    under_test.wait_for_empty_queue()
    assert under_test.cancel_all()
    
    # Act
    outcome1 = under_test.wait_for_execution(id="task1")
    outcome2 = under_test.wait_for_execution(id="task2")

    # Assert
    assert outcome1 is None
    assert outcome2 is None
    assert counter == 0
