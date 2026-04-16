# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import threading
from typing import Any

import pytest
from pydantic import Field
from workflows import (
    Context,
    FailureInfo,
    RetryInfo,
    StepFailedEvent,
    Workflow,
    catch_error,
    step,
)
from workflows.context.serializers import JsonSerializer
from workflows.errors import (
    ContextStateError,
    WorkflowRuntimeError,
    WorkflowValidationError,
)
from workflows.events import Event, StartEvent, StopEvent, WorkflowFailedEvent
from workflows.retry_policy import (
    retry_always,
    retry_policy,
    stop_after_attempt,
    wait_fixed,
)
from workflows.runtime.types.internal_state import BrokerState, EventAttempt


def _retry(attempts: int) -> Any:
    return retry_policy(
        retry=retry_always(),
        wait=wait_fixed(0),
        stop=stop_after_attempt(attempts),
    )


class _Marker(Event):
    """Stub event used in graph-validation tests."""


class _InputStart(StartEvent):
    query: str = "hello"


class _LockStart(StartEvent):
    lock: Any = Field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# retry_info()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_info_defaults_on_first_attempt() -> None:
    captured: dict[str, RetryInfo] = {}

    class Flow(Workflow):
        @step
        async def first(self, ctx: Context, ev: StartEvent) -> StopEvent:
            captured["info"] = ctx.retry_info()
            return StopEvent(result="ok")

    await Flow(timeout=5).run()
    info = captured["info"]
    assert info.attempt == 1
    assert info.elapsed_seconds == 0.0
    assert info.last_failure is None


@pytest.mark.asyncio
async def test_retry_info_after_failure_populated() -> None:
    observed: list[RetryInfo] = []

    class Flow(Workflow):
        @step(retry_policy=_retry(3))
        async def flaky(self, ctx: Context, ev: StartEvent) -> StopEvent:
            info = ctx.retry_info()
            observed.append(info)
            if info.attempt < 2:
                raise ValueError("boom")
            return StopEvent(result="ok")

    result = await Flow(timeout=5).run()
    assert result == "ok"
    assert observed[0].attempt == 1
    assert observed[0].last_failure is None
    assert observed[1].attempt == 2
    assert observed[1].elapsed_seconds >= 0.0
    assert observed[1].last_failure is not None
    assert observed[1].last_failure.exception_type == "builtins.ValueError"
    assert observed[1].last_failure.exception_message == "boom"
    assert "ValueError" in observed[1].last_failure.traceback


def test_retry_info_outside_step_raises() -> None:
    class Flow(Workflow):
        @step
        async def a(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="ok")

    ctx: Context = Context(Flow())
    with pytest.raises((WorkflowRuntimeError, ContextStateError)):
        ctx.retry_info()


def test_last_failure_serialization_roundtrip() -> None:
    class Flow(Workflow):
        @step
        async def a(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="ok")

    wf = Flow()
    state = BrokerState.from_workflow(wf)
    failure = FailureInfo(
        exception_type="builtins.ValueError",
        exception_message="boom",
        traceback="Traceback",
        failed_at=123.456,
    )
    state.workers["a"].queue.append(
        EventAttempt(
            event=StartEvent(),
            attempts=1,
            first_attempt_at=100.0,
            last_failure=failure,
        )
    )
    serialized = state.to_serialized(JsonSerializer())
    restored = BrokerState.from_serialized(serialized, wf, JsonSerializer())
    restored_attempt = restored.workers["a"].queue[0]
    assert restored_attempt.last_failure == failure


# ---------------------------------------------------------------------------
# Workflow construction validation
# ---------------------------------------------------------------------------


def test_multiple_catch_error_handlers_invalid() -> None:
    with pytest.raises(WorkflowValidationError, match="Only one @catch_error"):

        class MultiHandlerFlow(Workflow):
            @step
            async def a(self, ev: StartEvent) -> StopEvent:
                return StopEvent(result="ok")

            @catch_error
            async def h1(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
                return StopEvent(result="one")

            @catch_error
            async def h2(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
                return StopEvent(result="two")

        MultiHandlerFlow()


def test_catch_error_non_stop_return_invalid() -> None:
    with pytest.raises(WorkflowValidationError, match="must only return StopEvent"):

        class BadFlow(Workflow):
            @step
            async def a(self, ev: StartEvent) -> StopEvent:
                return StopEvent(result="ok")

            @catch_error
            async def handler(self, ctx: Context, ev: StepFailedEvent) -> _Marker:
                return _Marker()

        BadFlow()


def test_catch_error_wrong_event_type_invalid() -> None:
    with pytest.raises(WorkflowValidationError, match="must accept StepFailedEvent"):

        @catch_error
        async def bad_handler(self: Any, ctx: Context, ev: StartEvent) -> StopEvent:
            return StopEvent()


# ---------------------------------------------------------------------------
# Runtime routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_catch_error_returning_stop_completes_workflow() -> None:
    class Flow(Workflow):
        @step(retry_policy=_retry(1))
        async def flaky(self, ev: StartEvent) -> StopEvent:
            raise RuntimeError("transient")

        @catch_error
        async def handler(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
            return StopEvent(result={"recovered_from": ev.step_name})

    handler = Flow(timeout=5).run()
    events: list[Event] = []
    async for ev in handler.stream_events():
        events.append(ev)
    result = await handler
    assert result == {"recovered_from": "flaky"}
    assert not any(isinstance(ev, WorkflowFailedEvent) for ev in events)


@pytest.mark.asyncio
async def test_catch_error_raising_fails_workflow() -> None:
    class Flow(Workflow):
        @step(retry_policy=_retry(1))
        async def flaky(self, ev: StartEvent) -> StopEvent:
            raise RuntimeError("primary")

        @catch_error
        async def handler(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
            raise ValueError("handler-failed")

    handler_run = Flow(timeout=5).run()
    events: list[Event] = []
    async for ev in handler_run.stream_events():
        events.append(ev)
    with pytest.raises(ValueError, match="handler-failed"):
        await handler_run
    failed_events = [ev for ev in events if isinstance(ev, WorkflowFailedEvent)]
    assert len(failed_events) == 1
    assert failed_events[0].step_name == "handler"
    assert failed_events[0].exception_message == "handler-failed"


@pytest.mark.asyncio
async def test_catch_error_not_invoked_on_recoverable_retry() -> None:
    handler_invoked: list[bool] = []

    class Flow(Workflow):
        attempts = 0

        @step(retry_policy=_retry(3))
        async def flaky(self, ev: StartEvent) -> StopEvent:
            self.attempts += 1
            if self.attempts < 2:
                raise ValueError("transient")
            return StopEvent(result="recovered")

        @catch_error
        async def handler(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
            handler_invoked.append(True)
            return StopEvent(result="caught")

    result = await Flow(timeout=5).run()
    assert result == "recovered"
    assert handler_invoked == []


@pytest.mark.asyncio
async def test_step_failed_event_fields() -> None:
    captured: dict[str, StepFailedEvent] = {}

    class Flow(Workflow):
        @step(retry_policy=_retry(2))
        async def flaky(self, ev: _InputStart) -> StopEvent:
            raise RuntimeError(f"bad:{ev.query}")

        @catch_error
        async def handler(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
            captured["ev"] = ev
            return StopEvent(result="caught")

    await Flow(timeout=5).run(start_event=_InputStart(query="hello"))
    ev = captured["ev"]
    assert ev.step_name == "flaky"
    assert ev.input_event_type.endswith("._InputStart")
    assert ev.input_event.get("query") == "hello"
    assert ev.exception_type == "builtins.RuntimeError"
    assert ev.exception_message == "bad:hello"
    assert "RuntimeError" in ev.traceback
    assert ev.attempt == 2
    assert ev.elapsed_seconds >= 0.0


@pytest.mark.asyncio
async def test_step_failed_event_non_serializable_input() -> None:
    captured: dict[str, StepFailedEvent] = {}

    class Flow(Workflow):
        @step(retry_policy=_retry(1))
        async def flaky(self, ev: _LockStart) -> StopEvent:
            raise RuntimeError("boom")

        @catch_error
        async def handler(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
            captured["ev"] = ev
            return StopEvent(result="caught")

    await Flow(timeout=5).run(start_event=_LockStart())
    ev = captured["ev"]
    assert ev.input_event.get("__non_serializable") is True
    assert ev.input_event["type"].endswith("_LockStart")
    assert "_LockStart" in ev.input_event["repr"]


@pytest.mark.asyncio
async def test_catch_error_not_invoked_on_timeout() -> None:
    import asyncio

    handler_invoked: list[bool] = []

    class Flow(Workflow):
        @step
        async def slow(self, ev: StartEvent) -> StopEvent:
            await asyncio.sleep(5)
            return StopEvent(result="unreachable")

        @catch_error
        async def handler(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
            handler_invoked.append(True)
            return StopEvent(result="caught")

    from workflows.errors import WorkflowTimeoutError

    with pytest.raises(WorkflowTimeoutError):
        await Flow(timeout=0.1).run()
    assert handler_invoked == []


@pytest.mark.asyncio
async def test_baseline_without_catch_error_still_fails() -> None:
    class Flow(Workflow):
        @step(retry_policy=_retry(1))
        async def flaky(self, ev: StartEvent) -> StopEvent:
            raise ValueError("boom")

    handler = Flow(timeout=5).run()
    events: list[Event] = []
    async for ev in handler.stream_events():
        events.append(ev)
    with pytest.raises(ValueError, match="boom"):
        await handler
    assert any(isinstance(ev, WorkflowFailedEvent) for ev in events)


@pytest.mark.asyncio
async def test_catch_error_can_read_context_state() -> None:
    class Flow(Workflow):
        @step(retry_policy=_retry(1))
        async def flaky(self, ctx: Context, ev: StartEvent) -> StopEvent:
            await ctx.store.set("progress", "halfway")
            raise RuntimeError("boom")

        @catch_error
        async def handler(self, ctx: Context, ev: StepFailedEvent) -> StopEvent:
            progress = await ctx.store.get("progress", default="unset")
            return StopEvent(result={"progress": progress, "step": ev.step_name})

    result = await Flow(timeout=5).run()
    assert result == {"progress": "halfway", "step": "flaky"}
