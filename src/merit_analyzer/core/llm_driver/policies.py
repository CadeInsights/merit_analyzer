from enum import Enum


class AGENT(Enum):
    ERROR_ANALYZER = "error_analyzer"
    SUITE_SCHEMA_BUILDER = "suite_schema_builder"
    TEST = "test"


class TOOL(Enum):
    READ = "read"  # Return file contents without changing them.
    WRITE = "write"  # Replace an entire file with provided contents.
    EDIT = "edit"  # Apply targeted edits to a file in place.
    GLOB = "glob"  # List files matching a shell-style pattern.
    GREP = "grep"  # Search files for lines matching a pattern.
    BASH = "bash"  # Run commands in a persistent bash session.
    WEB_FETCH = "web_fetch"  # Download full content from a specific URL.
    WEB_SEARCH = "web_search"  # Execute a web search query with optional filters.
    TODO_WRITE = "todo_write"  # Create or update structured todo entries.
    BASH_OUTPUT = "bash_output"  # Retrieve buffered output from the bash session.
    KILL_BASH = "kill_bash"  # Terminate the persistent bash session.
    LIST_MCP_RESOURCES = "list_mcp_resources"  # List MCP resources provided by servers.
    READ_MCP_RESOURCE = "read_mcp_resource"  # Read a specific MCP resource's content.
    LS = "ls"  # Show directory contents without modifications.
    TASK = "task"  # Delegate work to a sub-agent with its own tools.
    SLASH_COMMAND = "slash_command"  # Invoke a custom slash command handler.


class FILE_ACCESS_POLICY(Enum):
    READ_ONLY = (
        TOOL.READ,
        TOOL.GREP,
        TOOL.GLOB,
        TOOL.LS,
        TOOL.WEB_SEARCH,
        TOOL.WEB_FETCH,
        TOOL.BASH_OUTPUT,
        TOOL.LIST_MCP_RESOURCES,
        TOOL.READ_MCP_RESOURCE,
    )
    READ_AND_WRITE = tuple(tool for tool in READ_ONLY) + (TOOL.EDIT, TOOL.WRITE, TOOL.TODO_WRITE)
    FULL_ACCESS = tuple(tool for tool in READ_AND_WRITE) + (
        TOOL.BASH,
        TOOL.SLASH_COMMAND,
        TOOL.KILL_BASH,
    )
    READ_AND_PLAN = tuple(tool for tool in READ_ONLY) + (TOOL.TODO_WRITE,)
