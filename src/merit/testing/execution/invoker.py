"""Test invocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from merit.testing.models import MeritTestDefinition


class TestInvoker(ABC):
    """Invokes test functions."""

    @abstractmethod
    async def invoke(self, definition: MeritTestDefinition, kwargs: dict[str, Any]) -> None:
        """Invoke the test function with the given arguments."""


@dataclass
class DefaultTestInvoker(TestInvoker):
    """Default test invoker that handles sync and async functions."""

    async def invoke(self, definition: MeritTestDefinition, kwargs: dict[str, Any]) -> None:
        """Invoke the test function with the given arguments."""
        fn = definition.fn
        if definition.is_async:
            await fn(**kwargs)
        else:
            fn(**kwargs)
