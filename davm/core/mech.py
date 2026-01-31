"""
DAVM - Digital Assistant Virtual Mech

The main orchestrator class that acts as the "mech suit" for LLM pilots.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Sequence

from davm.core.config import Settings, get_settings
from davm.pilots.base import BasePilot, GenerationResult, Message, ModelInfo, PilotInfo, ToolCallRequest
from davm.pilots.anthropic_pilot import AnthropicPilot
from davm.pilots.ollama_pilot import OllamaPilot
from davm.systems.memory import MemorySystem, MemorySearchResult
from davm.systems.filesystem import FileSystem, FileInfo, FileOperationResult
from davm.systems.web import WebSystem, WebPage, FetchResult
from davm.systems.tools import (
    Tool,
    ToolCall,
    ToolResult,
    ToolCategory,
    ToolExecutor,
    ALL_TOOLS,
    get_tools_by_category,
    get_tool_by_name,
)


AutonomyLevel = Literal["ask", "semi", "full"]
PilotType = Literal["anthropic", "ollama"]


@dataclass
class MechStatus:
    """Current status of the DAVM mech."""
    
    active: bool = False
    pilot_type: PilotType | None = None
    pilot_info: PilotInfo | None = None
    autonomy_level: AutonomyLevel = "semi"
    systems_active: list[str] = field(default_factory=list)
    memory_stats: dict[str, Any] | None = None


class DAVM:
    """
    Digital Assistant Virtual Mech - The main orchestrator.
    
    DAVM is like a "mech suit" that LLMs (pilots) can operate.
    It provides the pilot with access to various systems:
    - Memory (RAG database for long-term recall)
    - File System (read/write files)
    - Web (scraping and research)
    
    The pilot (LLM) can be swapped between different backends:
    - Anthropic Claude
    - Ollama (local models)
    
    Example usage:
        mech = DAVM()
        await mech.activate(pilot="anthropic", model="claude-sonnet-4-20250514")
        response = await mech.chat("Hello, how are you?")
        print(response.content)
        
    With memory enabled:
        mech = DAVM()
        await mech.activate(pilot="ollama")
        await mech.enable_memory()  # Enable persistent memory
        
        # This conversation will be remembered
        response = await mech.chat("My favorite color is blue.")
        
        # Later (even in a new session), the mech can recall this
        response = await mech.chat("What's my favorite color?")
        # Response will reference the stored memory about blue
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the DAVM mech.
        
        Args:
            settings: Optional settings override. If not provided,
                     loads from environment variables.
        """
        self._settings = settings or get_settings()
        self._pilot: BasePilot | None = None
        self._pilot_type: PilotType | None = None
        self._autonomy_level: AutonomyLevel = self._settings.davm_autonomy_level
        self._active = False
        
        # Memory system (initialized when enabled)
        self._memory_system: MemorySystem | None = None
        self._memory_enabled = False
        
        # File and Web systems
        self._file_system: FileSystem | None = None
        self._file_system_enabled = False
        self._web_system: WebSystem | None = None
        self._web_system_enabled = False
        
        # Tool executor (initialized lazily)
        self._tool_executor: ToolExecutor | None = None

        # Available pilots registry
        self._pilots: dict[PilotType, type[BasePilot]] = {
            "anthropic": AnthropicPilot,
            "ollama": OllamaPilot,
        }

    @property
    def active(self) -> bool:
        """Check if the mech is active (has a connected pilot)."""
        return self._active and self._pilot is not None and self._pilot.connected

    @property
    def pilot(self) -> BasePilot | None:
        """Get the current pilot."""
        return self._pilot

    @property
    def pilot_type(self) -> PilotType | None:
        """Get the current pilot type."""
        return self._pilot_type

    @property
    def autonomy_level(self) -> AutonomyLevel:
        """Get the current autonomy level."""
        return self._autonomy_level

    @property
    def memory_enabled(self) -> bool:
        """Check if memory system is enabled."""
        return self._memory_enabled and self._memory_system is not None

    @property
    def memory(self) -> MemorySystem | None:
        """Get the memory system (if enabled)."""
        return self._memory_system if self._memory_enabled else None

    @property
    def files_enabled(self) -> bool:
        """Check if file system is enabled."""
        return self._file_system_enabled and self._file_system is not None

    @property
    def files(self) -> FileSystem | None:
        """Get the file system (if enabled)."""
        return self._file_system if self._file_system_enabled else None

    @property
    def web_enabled(self) -> bool:
        """Check if web system is enabled."""
        return self._web_system_enabled and self._web_system is not None

    @property
    def web(self) -> WebSystem | None:
        """Get the web system (if enabled)."""
        return self._web_system if self._web_system_enabled else None

    def set_autonomy_level(self, level: AutonomyLevel) -> None:
        """
        Set the autonomy level for the mech.
        
        Args:
            level: "ask" (always ask), "semi" (ask for writes), "full" (autonomous)
        """
        self._autonomy_level = level

    async def enable_memory(self, persist_dir: str | None = None) -> None:
        """
        Enable the memory system for persistent recall.
        
        Args:
            persist_dir: Directory for memory storage. Defaults to settings.
        """
        if self._memory_enabled and self._memory_system:
            return  # Already enabled
        
        storage_dir = persist_dir or str(self._settings.davm_memory_dir)
        self._memory_system = MemorySystem(persist_dir=storage_dir)
        await self._memory_system.initialize()
        self._memory_system.start_conversation()
        self._memory_enabled = True

    async def disable_memory(self) -> None:
        """Disable the memory system."""
        if self._memory_system:
            await self._memory_system.shutdown()
        self._memory_system = None
        self._memory_enabled = False

    def enable_files(self, allowed_paths: Sequence[str | Path] | None = None) -> None:
        """
        Enable the file system for file operations.
        
        Args:
            allowed_paths: List of allowed paths. Defaults to settings.
        """
        if self._file_system_enabled and self._file_system:
            return  # Already enabled
        
        paths = list(allowed_paths) if allowed_paths else None
        self._file_system = FileSystem(
            settings=self._settings,
            allowed_paths=paths,
        )
        self._file_system_enabled = True

    def disable_files(self) -> None:
        """Disable the file system."""
        self._file_system = None
        self._file_system_enabled = False

    def enable_web(self, timeout: float = 30.0) -> None:
        """
        Enable the web system for web scraping.
        
        Args:
            timeout: Request timeout in seconds.
        """
        if self._web_system_enabled and self._web_system:
            return  # Already enabled
        
        self._web_system = WebSystem(timeout=timeout)
        self._web_system_enabled = True

    def disable_web(self) -> None:
        """Disable the web system."""
        self._web_system = None
        self._web_system_enabled = False

    async def activate(
        self,
        pilot: PilotType | None = None,
        model: str | None = None,
        enable_memory: bool = True,
        enable_files: bool = True,
        enable_web: bool = True,
    ) -> bool:
        """
        Activate the mech with a specific pilot.
        
        Args:
            pilot: Pilot type to use ("anthropic" or "ollama").
                  Defaults to setting from .env.
            model: Specific model to use. Defaults to pilot's default.
            enable_memory: Whether to enable memory system. Defaults to True.
            enable_files: Whether to enable file system. Defaults to True.
            enable_web: Whether to enable web system. Defaults to True.
            
        Returns:
            True if activation successful.
        """
        # Deactivate current pilot if any
        if self._active:
            await self.deactivate()

        # Determine which pilot to use
        pilot_type = pilot or self._settings.davm_default_pilot
        
        if pilot_type not in self._pilots:
            raise ValueError(f"Unknown pilot type: {pilot_type}")

        # Create and connect the pilot
        self._pilot = self._pilots[pilot_type]()
        
        try:
            await self._pilot.connect(model=model)
            self._pilot_type = pilot_type
            self._active = True
            
            # Enable systems by default
            if enable_memory:
                await self.enable_memory()
            if enable_files:
                self.enable_files()
            if enable_web:
                self.enable_web()
            
            return True
        except Exception as e:
            self._pilot = None
            self._pilot_type = None
            self._active = False
            raise RuntimeError(f"Failed to activate mech with {pilot_type} pilot: {e}")

    async def deactivate(self) -> None:
        """Deactivate the mech and disconnect the pilot."""
        # End current conversation in memory
        if self._memory_system:
            self._memory_system.end_conversation()
        
        if self._pilot:
            await self._pilot.disconnect()
        self._pilot = None
        self._pilot_type = None
        self._active = False

    async def switch_pilot(
        self,
        pilot: PilotType,
        model: str | None = None,
        preserve_history: bool = True,
    ) -> bool:
        """
        Switch to a different pilot.
        
        Args:
            pilot: New pilot type to use.
            model: Specific model to use with new pilot.
            preserve_history: Keep conversation history when switching.
            
        Returns:
            True if switch successful.
        """
        # Save current history if needed
        history = None
        if preserve_history and self._pilot:
            history = self._pilot.get_history_for_context()

        # Remember if memory was enabled
        memory_was_enabled = self._memory_enabled

        # Activate new pilot (but don't re-enable memory, we'll preserve it)
        # Temporarily disable memory to preserve the existing session
        old_memory = self._memory_system
        self._memory_system = None
        
        await self.activate(pilot=pilot, model=model, enable_memory=False)
        
        # Restore memory system
        self._memory_system = old_memory
        self._memory_enabled = memory_was_enabled

        # Restore history
        if history and self._pilot:
            for msg in history:
                self._pilot.add_message(msg.role, msg.content)

        return True

    async def chat(
        self,
        message: str,
        stream: bool = False,
        use_memory: bool = True,
    ) -> GenerationResult | AsyncGenerator[str, None]:
        """
        Send a message and get a response.
        
        If memory is enabled, this will:
        1. Search for relevant past memories
        2. Inject them into the context
        3. Generate a response
        4. Store the exchange as a new memory
        
        Args:
            message: The message to send.
            stream: If True, return an async generator for streaming response.
            use_memory: Whether to use memory for this message. Defaults to True.
            
        Returns:
            GenerationResult if stream=False, AsyncGenerator if stream=True.
        """
        if not self.active or not self._pilot:
            raise RuntimeError("Mech not active. Call activate() first.")

        # Get memory context if enabled
        memory_context = ""
        if use_memory and self._memory_enabled and self._memory_system:
            memory_context = await self._memory_system.get_context_for_prompt(
                current_prompt=message,
                max_memories=5,
                max_tokens=1500,
            )

        # Build the augmented prompt if we have memory context
        augmented_message = message
        if memory_context:
            augmented_message = f"{memory_context}\n\n[Current message:]\n{message}"

        if stream:
            # For streaming, we need to handle memory storage after
            return self._chat_stream_with_memory(message, augmented_message, use_memory)
        else:
            # Generate response
            result = await self._pilot.generate(augmented_message)
            
            # Store in memory
            if use_memory and self._memory_enabled and self._memory_system:
                await self._memory_system.store(
                    user_message=message,  # Store original message, not augmented
                    assistant_response=result.content,
                )
            
            return result

    async def _chat_stream_with_memory(
        self,
        original_message: str,
        augmented_message: str,
        use_memory: bool,
    ) -> AsyncGenerator[str, None]:
        """Internal helper for streaming with memory support."""
        full_response = ""
        
        async for chunk in self._pilot.generate_stream(augmented_message):
            full_response += chunk
            yield chunk
        
        # Store in memory after streaming completes
        if use_memory and self._memory_enabled and self._memory_system:
            await self._memory_system.store(
                user_message=original_message,
                assistant_response=full_response,
            )

    async def chat_stream(self, message: str, use_memory: bool = True) -> AsyncGenerator[str, None]:
        """
        Send a message and stream the response.
        
        Convenience method for streaming responses.
        
        Args:
            message: The message to send.
            use_memory: Whether to use memory for this message.
            
        Yields:
            Chunks of the response as they're generated.
        """
        if not self.active or not self._pilot:
            raise RuntimeError("Mech not active. Call activate() first.")

        # Get memory context if enabled
        memory_context = ""
        if use_memory and self._memory_enabled and self._memory_system:
            memory_context = await self._memory_system.get_context_for_prompt(
                current_prompt=message,
                max_memories=5,
                max_tokens=1500,
            )

        augmented_message = message
        if memory_context:
            augmented_message = f"{memory_context}\n\n[Current message:]\n{message}"

        full_response = ""
        async for chunk in self._pilot.generate_stream(augmented_message):
            full_response += chunk
            yield chunk

        # Store in memory after streaming completes
        if use_memory and self._memory_enabled and self._memory_system:
            await self._memory_system.store(
                user_message=message,
                assistant_response=full_response,
            )

    def _get_tool_executor(self) -> ToolExecutor:
        """Get or create the tool executor."""
        if self._tool_executor is None:
            self._tool_executor = ToolExecutor(self)
        return self._tool_executor

    def _get_allowed_tools(self) -> list[Tool]:
        """Get tools allowed based on autonomy level."""
        if self._autonomy_level == "full":
            # All tools available
            return ALL_TOOLS
        elif self._autonomy_level == "semi":
            # Read operations + memory, no destructive file ops
            return get_tools_by_category([
                ToolCategory.MEMORY,
                ToolCategory.FILES_READ,
                ToolCategory.WEB,
            ])
        else:  # "ask"
            # Only memory for now in ask mode
            return get_tools_by_category([ToolCategory.MEMORY])

    def _get_tools_schema(self) -> list[dict]:
        """Get tool schemas for the current pilot."""
        tools = self._get_allowed_tools()
        
        if self._pilot_type == "anthropic":
            return [t.to_anthropic_schema() for t in tools]
        else:  # ollama
            return [t.to_ollama_schema() for t in tools]

    def _format_tool_results_for_llm(self, tool_results: list[ToolResult]) -> str:
        """Format tool results into a message for the LLM to process."""
        if not tool_results:
            return "No tool results."
        
        parts = ["Here are the results from the tools I used:"]
        for result in tool_results:
            if result.success:
                parts.append(f"\n[{result.tool_call_id}] Success:")
                parts.append(result.to_string())
            else:
                parts.append(f"\n[{result.tool_call_id}] Error: {result.error}")
        
        parts.append("\nPlease provide a helpful response based on these results.")
        return "\n".join(parts)

    async def chat_with_tools(
        self,
        message: str,
        use_memory: bool = True,
        max_tool_rounds: int = 5,
    ) -> tuple[str, list[ToolResult]]:
        """
        Send a message and let the LLM use tools to answer.
        
        This is the agentic mode where the LLM can autonomously
        use the mech's capabilities (memory, files, web) to help.
        
        Args:
            message: The user's message.
            use_memory: Whether to use memory context.
            max_tool_rounds: Maximum rounds of tool use.
            
        Returns:
            Tuple of (final_response, list_of_tool_results)
        """
        if not self.active or not self._pilot:
            raise RuntimeError("Mech not active. Call activate() first.")

        # Get memory context if enabled
        memory_context = ""
        if use_memory and self._memory_enabled and self._memory_system:
            memory_context = await self._memory_system.get_context_for_prompt(
                current_prompt=message,
                max_memories=5,
                max_tokens=1500,
            )

        augmented_message = message
        if memory_context:
            augmented_message = f"{memory_context}\n\n[Current message:]\n{message}"

        # Get tools schema
        tools_schema = self._get_tools_schema()
        
        if not tools_schema:
            # No tools available, just do normal chat
            result = await self._pilot.generate(augmented_message)
            return result.content, []

        executor = self._get_tool_executor()
        all_tool_results: list[ToolResult] = []
        final_content = ""
        last_tool_results: list[ToolResult] = []
        
        # Tool use loop - LLM calls tools, we execute them, send results back
        current_prompt = augmented_message
        for round_num in range(max_tool_rounds):
            result = await self._pilot.generate_with_tools(
                prompt=current_prompt,
                tools=tools_schema,
            )
            
            # If no tool calls, we're done - use the content as final response
            if not result.tool_calls:
                final_content = result.content
                break
            
            # Execute tool calls
            round_tool_results: list[ToolResult] = []
            for tc in result.tool_calls:
                tool_call = ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                )
                
                # Check if tool requires confirmation in semi mode
                tool = get_tool_by_name(tc.name)
                if tool and tool.requires_confirmation and self._autonomy_level != "full":
                    # Skip destructive operations in non-full autonomy
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        success=False,
                        result=None,
                        error=f"Tool '{tc.name}' requires confirmation. Set autonomy to 'full' or use the CLI.",
                    )
                else:
                    tool_result = await executor.execute(tool_call)
                
                round_tool_results.append(tool_result)
                all_tool_results.append(tool_result)
            
            # Prepare tool results for next round
            last_tool_results = round_tool_results
            current_prompt = self._format_tool_results_for_llm(round_tool_results)
        else:
            # Loop exhausted max rounds - force a final response from tool results
            if last_tool_results:
                tool_results_text = self._format_tool_results_for_llm(last_tool_results)
                final_result = await self._pilot.generate(
                    f"Based on the following tool results, provide a helpful response to the user:\n\n{tool_results_text}"
                )
                final_content = final_result.content
            else:
                final_content = "I tried to help but couldn't complete the task."

        # Store in memory
        if use_memory and self._memory_enabled and self._memory_system:
            # Include tool usage summary in what we store
            tool_summary = ""
            if all_tool_results:
                tool_names = [r.tool_call_id.split("_")[0] for r in all_tool_results if "_" in r.tool_call_id]
                if not tool_names:
                    tool_names = ["tools"]
                tool_summary = f" [Used: {', '.join(set(tool_names))}]"
            
            await self._memory_system.store(
                user_message=message,
                assistant_response=final_content + tool_summary,
            )

        return final_content, all_tool_results

    async def search_memory(
        self,
        query: str,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """
        Search through stored memories.
        
        Args:
            query: What to search for.
            limit: Maximum results to return.
            
        Returns:
            List of matching memories with relevance scores.
        """
        if not self._memory_enabled or not self._memory_system:
            return []
        
        return await self._memory_system.search(query, limit=limit)

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about the memory system."""
        if not self._memory_system:
            return {"enabled": False}
        
        stats = await self._memory_system.get_stats()
        stats["enabled"] = self._memory_enabled
        return stats

    async def clear_memory(self) -> int:
        """
        Clear all stored memories.
        
        Returns:
            Number of memories deleted.
        """
        if not self._memory_system:
            return 0
        return await self._memory_system.clear_all()

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the current pilot."""
        if self._pilot:
            self._pilot.set_system_prompt(prompt)

    def clear_history(self) -> None:
        """Clear the conversation history (not long-term memory)."""
        if self._pilot:
            self._pilot.clear_history()

    def get_history(self) -> list[Message]:
        """Get the conversation history."""
        if self._pilot:
            return self._pilot.conversation_history
        return []

    async def list_available_models(
        self,
        pilot: PilotType | None = None,
    ) -> list[ModelInfo]:
        """
        List available models for a pilot.
        
        Args:
            pilot: Pilot type to query. If None, uses current pilot.
            
        Returns:
            List of available models.
        """
        if pilot is None:
            if not self._pilot:
                raise RuntimeError("No pilot active and no pilot type specified.")
            return await self._pilot.list_models()

        # Create temporary pilot instance to query models
        temp_pilot = self._pilots[pilot]()
        try:
            await temp_pilot.connect()
            models = await temp_pilot.list_models()
            await temp_pilot.disconnect()
            return models
        except Exception:
            # For Anthropic, we can return the static list even without connecting
            if pilot == "anthropic":
                return await temp_pilot.list_models()
            raise

    async def get_status(self) -> MechStatus:
        """Get the current status of the mech."""
        systems = []
        if self._memory_enabled:
            systems.append("memory")
        if self._file_system:
            systems.append("filesystem")
        if self._web_system:
            systems.append("web")

        memory_stats = None
        if self._memory_system:
            memory_stats = await self._memory_system.get_stats()

        return MechStatus(
            active=self.active,
            pilot_type=self._pilot_type,
            pilot_info=self._pilot.get_info() if self._pilot else None,
            autonomy_level=self._autonomy_level,
            systems_active=systems,
            memory_stats=memory_stats,
        )

    def get_pilot_info(self) -> PilotInfo | None:
        """Get information about the current pilot."""
        if self._pilot:
            return self._pilot.get_info()
        return None

    def start_new_conversation(self) -> str | None:
        """
        Start a new conversation session in memory.
        
        This creates a new conversation ID for grouping related memories.
        
        Returns:
            The new conversation ID, or None if memory not enabled.
        """
        if self._memory_system:
            return self._memory_system.start_conversation()
        return None

    # === File System Convenience Methods ===

    def read_file(self, path: str) -> FileOperationResult:
        """
        Read a file's contents.
        
        Args:
            path: Path to the file.
            
        Returns:
            FileOperationResult with contents in .data
        """
        if not self._file_system:
            return FileOperationResult(
                success=False,
                message="File system not enabled. Call enable_files() first.",
            )
        return self._file_system.read_file(path)

    def write_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
    ) -> FileOperationResult:
        """
        Write content to a file.
        
        Args:
            path: Path to the file.
            content: Content to write.
            overwrite: If True, overwrite existing file.
            
        Returns:
            FileOperationResult indicating success.
        """
        if not self._file_system:
            return FileOperationResult(
                success=False,
                message="File system not enabled. Call enable_files() first.",
            )
        return self._file_system.write_file(path, content, overwrite=overwrite)

    def list_files(self, path: str, pattern: str | None = None) -> list[FileInfo]:
        """
        List files in a directory.
        
        Args:
            path: Directory path.
            pattern: Optional glob pattern.
            
        Returns:
            List of FileInfo objects.
        """
        if not self._file_system:
            return []
        return self._file_system.list_dir(path, pattern=pattern)

    def search_files(self, pattern: str, path: str | None = None) -> list[FileInfo]:
        """
        Search for files matching a pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py").
            path: Starting directory. If None, searches all allowed paths.
            
        Returns:
            List of matching FileInfo objects.
        """
        if not self._file_system:
            return []
        return self._file_system.search(pattern, start_path=path)

    def get_file_tree(self, path: str, max_depth: int = 3) -> str:
        """
        Get a tree representation of a directory.
        
        Args:
            path: Root directory.
            max_depth: Maximum depth to traverse.
            
        Returns:
            String tree representation.
        """
        if not self._file_system:
            return "File system not enabled."
        return self._file_system.get_tree(path, max_depth=max_depth)

    # === Web System Convenience Methods ===

    async def fetch_url(self, url: str) -> FetchResult:
        """
        Fetch a web page.
        
        Args:
            url: URL to fetch.
            
        Returns:
            FetchResult with page content.
        """
        if not self._web_system:
            return FetchResult(
                success=False,
                message="Web system not enabled. Call enable_web() first.",
            )
        return await self._web_system.fetch(url)

    async def fetch_text(self, url: str) -> str:
        """
        Fetch a URL and return just the text content.
        
        Args:
            url: URL to fetch.
            
        Returns:
            Text content or error message.
        """
        if not self._web_system:
            return "Web system not enabled. Call enable_web() first."
        return await self._web_system.fetch_text(url)

    async def get_page_summary(self, url: str) -> dict:
        """
        Get a summary of a web page.
        
        Args:
            url: URL to fetch.
            
        Returns:
            Dictionary with page summary.
        """
        if not self._web_system:
            return {"success": False, "error": "Web system not enabled."}
        return await self._web_system.get_page_summary(url)

    # Context manager support for clean activation/deactivation
    async def __aenter__(self) -> "DAVM":
        """Enter async context - activate the mech."""
        await self.activate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - deactivate the mech."""
        await self.deactivate()
