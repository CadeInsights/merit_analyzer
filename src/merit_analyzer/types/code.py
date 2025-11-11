from enum import Enum

from pydantic import BaseModel, Field


class ComponentType(str, Enum):
    """Supported component types referenced in reports."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"
    TEST = "test"


class CodeComponent(BaseModel):
    """Code element linked to a failure."""

    name: str = Field(..., description="Display name for the element")
    path: str = Field(..., description="Path to the file that owns the element")
    type: ComponentType = Field(..., description="Kind of the code component")
