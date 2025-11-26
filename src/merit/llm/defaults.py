"""Default constants for LLM agents and tools."""

from enum import Enum


MAX_JSON_PARSING_ATTEMPTS = 2
MAX_AGENT_TURNS = 10


class Tool(Enum):
    """Available tools for agents."""

    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    GLOB = "glob"
    GREP = "grep"
    BASH = "bash"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    TODO_WRITE = "todo_write"
    BASH_OUTPUT = "bash_output"
    KILL_BASH = "kill_bash"
    LIST_MCP_RESOURCES = "list_mcp_resources"
    READ_MCP_RESOURCE = "read_mcp_resource"
    LS = "ls"
    TASK = "task"
    SLASH_COMMAND = "slash_command"


class FileAccessLevel(Enum):
    """File access policies defining which tools are permitted."""

    READ_ONLY = (
        Tool.READ,
        Tool.GREP,
        Tool.GLOB,
        Tool.LS,
        Tool.WEB_SEARCH,
        Tool.WEB_FETCH,
        Tool.BASH_OUTPUT,
        Tool.LIST_MCP_RESOURCES,
        Tool.READ_MCP_RESOURCE,
    )
    READ_AND_WRITE = READ_ONLY + (Tool.EDIT, Tool.WRITE, Tool.TODO_WRITE)
    FULL_ACCESS = READ_AND_WRITE + (Tool.BASH, Tool.SLASH_COMMAND, Tool.KILL_BASH)
    READ_AND_PLAN = READ_ONLY + (Tool.TODO_WRITE,)
