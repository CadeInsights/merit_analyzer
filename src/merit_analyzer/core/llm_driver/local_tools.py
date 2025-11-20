"""Tools for interacting with local files in shell"""

# NOTE: Tools for writing and editing files currently implemented in a naive way.
# They must be refactored before allowing agents to perform relevant operations
# with files.

import re
from pathlib import Path
from typing import Any


def read(file_path: str, offset: int | None = None, limit: int | None = None) -> dict[str, Any]:
    """Return numbered file contents with optional slicing."""
    lines = Path(file_path).read_text().splitlines()
    start = max(offset - 1, 0) if offset else 0
    stop = start + limit if limit else None
    view = lines[start:stop]
    content = "\n".join(f"{index}:{text}" for index, text in enumerate(view, start=start + 1))
    return {"content": content, "total_lines": len(lines), "lines_returned": len(view)}


def write(file_path: str, content: str) -> dict[str, Any]:
    """Overwrite a file and report bytes written."""
    written = Path(file_path).write_text(content)
    return {"message": "file updated", "bytes_written": written, "file_path": file_path}


def edit(file_path: str, old_string: str, new_string: str, replace_all: bool | None = None) -> dict[str, Any]:
    """Replace occurrences of a string within a file."""
    path = Path(file_path)
    original = path.read_text()
    total = original.count(old_string)
    if total == 0:
        return {"message": "no matches found", "replacements": 0, "file_path": file_path}
    count = total if replace_all else 1
    updated = original.replace(old_string, new_string, count)
    path.write_text(updated)
    replacements = total if replace_all else 1
    return {"message": "text replaced", "replacements": replacements, "file_path": file_path}


def glob(pattern: str, path: str | None = None) -> dict[str, Any]:
    """Return paths that match a glob pattern."""
    base = Path(path) if path else Path()
    matches = sorted(str(match) for match in base.glob(pattern))
    return {"matches": matches, "count": len(matches), "search_path": str(base)}


def grep(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    ignore_case: bool | None = None,
    show_line_numbers: bool | None = None,
    head_limit: int | None = None,
) -> dict[str, Any]:
    """Search files for lines matching a pattern."""
    base = Path(path) if path else Path()
    flags = re.MULTILINE | (re.IGNORECASE if ignore_case else 0)
    matcher = re.compile(pattern, flags)
    candidates = [base] if base.is_file() else [p for p in base.rglob(glob or "*") if p.is_file()]
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        for index, line in enumerate(candidate.read_text().splitlines(), start=1):
            if matcher.search(line):
                record: dict[str, Any] = {"file": str(candidate), "line": line}
                if show_line_numbers:
                    record["line_number"] = index
                results.append(record)
                if head_limit and len(results) >= head_limit:
                    return {"matches": results, "total_matches": len(results)}
    return {"matches": results, "total_matches": len(results)}


def ls(path: str | None = None) -> dict[str, Any]:
    """List directory entries."""
    target = Path(path) if path else Path()
    entries = sorted(str(entry) for entry in target.iterdir())
    return {"path": str(target), "entries": entries, "count": len(entries)}


def todo(todos: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize todo items by status."""
    totals = {"pending": 0, "in_progress": 0, "completed": 0}
    for todo in todos:
        status = todo.get("status", "pending")
        if status not in totals:
            continue
        totals[status] += 1
    total = sum(totals.values())
    return {"message": "todos recorded", "stats": {"total": total, **totals}}
