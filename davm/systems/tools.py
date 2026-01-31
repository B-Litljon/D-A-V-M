"""
DAVM Tools System - Function calling and tool use for LLM agents.

This system enables the LLM pilot to autonomously use the mech's capabilities
(memory, files, web) through structured tool calls.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable
import json


class ToolCategory(Enum):
    """Categories of tools for permission grouping."""
    MEMORY = "memory"
    FILES_READ = "files_read"
    FILES_WRITE = "files_write"
    WEB = "web"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class Tool:
    """Definition of a tool that the LLM can use."""
    
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter] = field(default_factory=list)
    requires_confirmation: bool = False  # For destructive operations
    
    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def to_ollama_schema(self) -> dict:
        """Convert to Ollama tool schema format (OpenAI-compatible)."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


@dataclass
class ToolCall:
    """A request from the LLM to use a tool."""
    
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""
    
    tool_call_id: str
    success: bool
    result: Any
    error: str | None = None
    
    def to_string(self) -> str:
        """Convert result to a string for the LLM."""
        if not self.success:
            return f"Error: {self.error}"
        
        if isinstance(self.result, str):
            return self.result
        elif isinstance(self.result, (dict, list)):
            return json.dumps(self.result, indent=2, default=str)
        else:
            return str(self.result)


# === Tool Definitions ===

MEMORY_TOOLS = [
    Tool(
        name="search_memory",
        description="Search through stored memories for relevant past conversations. Use this to recall what the user has told you before.",
        category=ToolCategory.MEMORY,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="What to search for in memories (e.g., 'favorite color', 'their job', 'project they mentioned')",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of memories to return",
                required=False,
                default=5,
            ),
        ],
    ),
    Tool(
        name="get_recent_memories",
        description="Get the most recent memories/conversations.",
        category=ToolCategory.MEMORY,
        parameters=[
            ToolParameter(
                name="limit",
                type="integer",
                description="Number of recent memories to retrieve",
                required=False,
                default=5,
            ),
        ],
    ),
]

FILE_READ_TOOLS = [
    Tool(
        name="read_file",
        description="Read the contents of a file. Use this when the user asks you to look at or analyze a file.",
        category=ToolCategory.FILES_READ,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file (can use ~ for home directory)",
            ),
        ],
    ),
    Tool(
        name="list_directory",
        description="List files and folders in a directory.",
        category=ToolCategory.FILES_READ,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Directory path to list (can use ~ for home directory)",
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="Optional glob pattern to filter results (e.g., '*.py')",
                required=False,
            ),
        ],
    ),
    Tool(
        name="search_files",
        description="Search for files matching a pattern.",
        category=ToolCategory.FILES_READ,
        parameters=[
            ToolParameter(
                name="pattern",
                type="string",
                description="Glob pattern to search for (e.g., '*.txt', '**/*.py')",
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Starting directory for search",
                required=False,
            ),
        ],
    ),
    Tool(
        name="get_file_info",
        description="Get information about a file (size, modified date, etc.).",
        category=ToolCategory.FILES_READ,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file",
            ),
        ],
    ),
]

FILE_WRITE_TOOLS = [
    Tool(
        name="write_file",
        description="Write content to a file. Creates the file if it doesn't exist.",
        category=ToolCategory.FILES_WRITE,
        requires_confirmation=True,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file",
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file",
            ),
            ToolParameter(
                name="overwrite",
                type="boolean",
                description="Whether to overwrite if file exists",
                required=False,
                default=False,
            ),
        ],
    ),
    Tool(
        name="create_directory",
        description="Create a new directory.",
        category=ToolCategory.FILES_WRITE,
        requires_confirmation=True,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path for the new directory",
            ),
        ],
    ),
    Tool(
        name="delete_file",
        description="Delete a file or empty directory.",
        category=ToolCategory.FILES_WRITE,
        requires_confirmation=True,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to delete",
            ),
        ],
    ),
    Tool(
        name="move_file",
        description="Move or rename a file.",
        category=ToolCategory.FILES_WRITE,
        requires_confirmation=True,
        parameters=[
            ToolParameter(
                name="source",
                type="string",
                description="Source path",
            ),
            ToolParameter(
                name="destination",
                type="string",
                description="Destination path",
            ),
        ],
    ),
]

WEB_TOOLS = [
    Tool(
        name="fetch_webpage",
        description="Fetch and read a webpage. Use this to look up information online or when the user shares a URL.",
        category=ToolCategory.WEB,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="URL to fetch",
            ),
        ],
    ),
    Tool(
        name="get_webpage_summary",
        description="Get a quick summary of a webpage (title, description, link count).",
        category=ToolCategory.WEB,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="URL to summarize",
            ),
        ],
    ),
    Tool(
        name="extract_links",
        description="Extract all links from a webpage.",
        category=ToolCategory.WEB,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="URL to extract links from",
            ),
        ],
    ),
]

# All tools combined
ALL_TOOLS = MEMORY_TOOLS + FILE_READ_TOOLS + FILE_WRITE_TOOLS + WEB_TOOLS


def get_tools_by_category(categories: list[ToolCategory]) -> list[Tool]:
    """Get tools filtered by categories."""
    return [t for t in ALL_TOOLS if t.category in categories]


def get_tool_by_name(name: str) -> Tool | None:
    """Get a tool by its name."""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


class ToolExecutor:
    """
    Executes tool calls using DAVM's systems.
    
    This class bridges tool calls from the LLM to actual
    DAVM system operations.
    """
    
    def __init__(self, davm: "DAVM"):
        """
        Initialize the tool executor.
        
        Args:
            davm: The DAVM instance to use for operations.
        """
        self._davm = davm
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register tool handlers."""
        # Memory tools
        self._handlers["search_memory"] = self._handle_search_memory
        self._handlers["get_recent_memories"] = self._handle_recent_memories
        
        # File read tools
        self._handlers["read_file"] = self._handle_read_file
        self._handlers["list_directory"] = self._handle_list_directory
        self._handlers["search_files"] = self._handle_search_files
        self._handlers["get_file_info"] = self._handle_get_file_info
        
        # File write tools
        self._handlers["write_file"] = self._handle_write_file
        self._handlers["create_directory"] = self._handle_create_directory
        self._handlers["delete_file"] = self._handle_delete_file
        self._handlers["move_file"] = self._handle_move_file
        
        # Web tools
        self._handlers["fetch_webpage"] = self._handle_fetch_webpage
        self._handlers["get_webpage_summary"] = self._handle_webpage_summary
        self._handlers["extract_links"] = self._handle_extract_links
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.
        
        Args:
            tool_call: The tool call to execute.
            
        Returns:
            ToolResult with the outcome.
        """
        handler = self._handlers.get(tool_call.name)
        
        if not handler:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_call.name}",
            )
        
        try:
            result = await handler(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                result=result,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                result=None,
                error=str(e),
            )
    
    # === Memory Handlers ===
    
    async def _handle_search_memory(self, query: str, limit: int = 5) -> str:
        """Search memories."""
        if not self._davm.memory_enabled:
            return "Memory system is not enabled."
        
        results = await self._davm.search_memory(query, limit=limit)
        
        if not results:
            return "No relevant memories found."
        
        output = []
        for r in results:
            output.append(f"[Relevance: {r.relevance_score:.0%}]")
            output.append(f"Time: {r.memory.timestamp.strftime('%Y-%m-%d %H:%M')}")
            output.append(r.memory.content)
            output.append("---")
        
        return "\n".join(output)
    
    async def _handle_recent_memories(self, limit: int = 5) -> str:
        """Get recent memories."""
        if not self._davm.memory_enabled or not self._davm.memory:
            return "Memory system is not enabled."
        
        memories = await self._davm.memory.get_recent_memories(limit=limit)
        
        if not memories:
            return "No memories stored yet."
        
        output = []
        for mem in memories:
            output.append(f"Time: {mem.timestamp.strftime('%Y-%m-%d %H:%M')}")
            output.append(mem.content)
            output.append("---")
        
        return "\n".join(output)
    
    # === File Read Handlers ===
    
    async def _handle_read_file(self, path: str) -> str:
        """Read a file."""
        if not self._davm.files_enabled:
            return "File system is not enabled."
        
        result = self._davm.read_file(path)
        
        if not result.success:
            return f"Error: {result.message}"
        
        content = result.data
        if isinstance(content, bytes):
            return f"[Binary file: {len(content)} bytes]"
        
        # Truncate very long files
        if len(content) > 10000:
            return content[:10000] + f"\n\n[Truncated - {len(content)} total characters]"
        
        return content
    
    async def _handle_list_directory(self, path: str, pattern: str | None = None) -> str:
        """List directory contents."""
        if not self._davm.files_enabled:
            return "File system is not enabled."
        
        files = self._davm.list_files(path, pattern=pattern)
        
        if not files:
            return f"No files found in {path}"
        
        output = [f"Contents of {path}:"]
        for f in files[:50]:
            if f.is_dir:
                output.append(f"  [DIR] {f.name}/")
            else:
                output.append(f"  {f.name} ({f.size_human})")
        
        if len(files) > 50:
            output.append(f"  ... and {len(files) - 50} more items")
        
        return "\n".join(output)
    
    async def _handle_search_files(self, pattern: str, path: str | None = None) -> str:
        """Search for files."""
        if not self._davm.files_enabled:
            return "File system is not enabled."
        
        results = self._davm.search_files(pattern, path)
        
        if not results:
            return f"No files matching '{pattern}' found."
        
        output = [f"Found {len(results)} files matching '{pattern}':"]
        for f in results[:30]:
            output.append(f"  {f.path}")
        
        if len(results) > 30:
            output.append(f"  ... and {len(results) - 30} more")
        
        return "\n".join(output)
    
    async def _handle_get_file_info(self, path: str) -> dict:
        """Get file info."""
        if not self._davm.files_enabled or not self._davm.files:
            return {"error": "File system is not enabled."}
        
        info = self._davm.files.get_file_info(path)
        
        if not info:
            return {"error": f"Cannot access {path}"}
        
        return {
            "name": info.name,
            "path": str(info.path),
            "is_directory": info.is_dir,
            "size": info.size_human,
            "modified": info.modified.isoformat(),
            "extension": info.extension,
        }
    
    # === File Write Handlers ===
    
    async def _handle_write_file(
        self, 
        path: str, 
        content: str, 
        overwrite: bool = False,
    ) -> str:
        """Write to a file."""
        if not self._davm.files_enabled:
            return "File system is not enabled."
        
        result = self._davm.write_file(path, content, overwrite=overwrite)
        return result.message
    
    async def _handle_create_directory(self, path: str) -> str:
        """Create a directory."""
        if not self._davm.files_enabled or not self._davm.files:
            return "File system is not enabled."
        
        result = self._davm.files.create_directory(path)
        return result.message
    
    async def _handle_delete_file(self, path: str) -> str:
        """Delete a file."""
        if not self._davm.files_enabled or not self._davm.files:
            return "File system is not enabled."
        
        result = self._davm.files.delete(path)
        return result.message
    
    async def _handle_move_file(self, source: str, destination: str) -> str:
        """Move a file."""
        if not self._davm.files_enabled or not self._davm.files:
            return "File system is not enabled."
        
        result = self._davm.files.move(source, destination)
        return result.message
    
    # === Web Handlers ===
    
    async def _handle_fetch_webpage(self, url: str) -> str:
        """Fetch a webpage."""
        if not self._davm.web_enabled:
            return "Web system is not enabled."
        
        result = await self._davm.fetch_url(url)
        
        if not result.success:
            return f"Error fetching {url}: {result.message}"
        
        if not result.page:
            return "No content retrieved."
        
        page = result.page
        output = [
            f"Title: {page.title or 'No title'}",
            f"URL: {page.url}",
            "",
            "Content:",
            page.text[:5000] if len(page.text) > 5000 else page.text,
        ]
        
        if len(page.text) > 5000:
            output.append(f"\n[Truncated - {len(page.text)} total characters]")
        
        return "\n".join(output)
    
    async def _handle_webpage_summary(self, url: str) -> dict:
        """Get webpage summary."""
        if not self._davm.web_enabled:
            return {"error": "Web system is not enabled."}
        
        return await self._davm.get_page_summary(url)
    
    async def _handle_extract_links(self, url: str) -> str:
        """Extract links from a webpage."""
        if not self._davm.web_enabled or not self._davm.web:
            return "Web system is not enabled."
        
        links = await self._davm.web.fetch_links(url)
        
        if not links:
            return f"No links found on {url}"
        
        output = [f"Links from {url}:"]
        for link in links[:30]:
            output.append(f"  [{link.text[:50]}]({link.url})")
        
        if len(links) > 30:
            output.append(f"  ... and {len(links) - 30} more links")
        
        return "\n".join(output)


# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from davm.core.mech import DAVM
