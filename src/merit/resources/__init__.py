"""Resource injection system."""

from merit.resources.resolver import (
    ResourceDef,
    ResourceResolver,
    Scope,
    clear_registry,
    get_registry,
    resource,
)


__all__ = [
    "ResourceDef",
    "ResourceResolver",
    "Scope",
    "clear_registry",
    "get_registry",
    "resource",
]
