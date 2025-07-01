import pytest
import time
from services.internal.action_manager import ActionManager, ActionResult
import logging

logger = logging.getLogger(__name__)

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