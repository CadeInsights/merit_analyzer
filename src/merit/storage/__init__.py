"""Storage module for persisting Merit test runs."""

from merit.storage.base import Store
from merit.storage.sqlite import SQLiteStore


__all__ = ["SQLiteStore", "Store"]
