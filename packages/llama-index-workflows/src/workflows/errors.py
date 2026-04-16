# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from dataclasses import dataclass


class WorkflowValidationError(Exception):
    """Raised when the workflow configuration or step signatures are invalid."""


class WorkflowTimeoutError(Exception):
    """Raised when a workflow run exceeds the configured timeout."""


class WorkflowRuntimeError(Exception):
    """Raised for runtime errors during step execution or event routing."""


class WorkflowDone(Exception):
    """Internal control-flow exception used to terminate workers at run end."""


class WorkflowCancelledByUser(Exception):
    """Raised when a run is cancelled via the handler or programmatically."""


class WorkflowStepDoesNotExistError(Exception):
    """Raised when addressing a step that does not exist in the workflow."""


class WorkflowConfigurationError(Exception):
    """Raised when a logical configuration error is detected pre-run."""


class ContextSerdeError(Exception):
    """Raised when serializing/deserializing a `Context` fails."""


class ContextStateError(Exception):
    """Raised when a context method is called in the wrong state.

    Context transitions between three states:
    - PreContext: Before workflow starts (configuration)
    - ExternalContext: During run, for handler/external code
    - InternalContext: During run, for step execution
    """


@dataclass(frozen=True)
class FailureInfo:
    """Describes a single failed step attempt.

    Attributes:
        exception_type: Fully qualified module + qualname of the exception class.
        exception_message: `str(exception)` of the raised exception.
        traceback: Joined output of `traceback.format_exception(...)`.
        failed_at: Unix timestamp when the failure occurred.
    """

    exception_type: str
    exception_message: str
    traceback: str
    failed_at: float


@dataclass(frozen=True)
class RetryInfo:
    """Snapshot of the currently-executing step's retry state.

    Returned by `Context.retry_info()`. On the first attempt `attempt` is 1,
    `elapsed_seconds` is 0.0 and `last_failure` is `None`. On subsequent
    retries `last_failure` describes the most recent prior failure.

    Attributes:
        attempt: 1-based attempt number of the currently-executing step.
        elapsed_seconds: Seconds since the first attempt began.
        last_failure: Information about the most recent prior failure, or None.
    """

    attempt: int
    elapsed_seconds: float
    last_failure: FailureInfo | None
