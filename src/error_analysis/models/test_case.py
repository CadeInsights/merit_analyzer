from dataclasses import dataclass
from typing import Any

@dataclass
class TestCase:
    input_value: Any
    expected: Any