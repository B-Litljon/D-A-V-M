"""Base class for LLM pilots."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator


@dataclass
class ModelInfo:
    """Information about a specific model."""

    name: str
    size: str | None = None
    description: str | None = None
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=list)


@dataclass
class PilotInfo:
    """Information about a pilot backend."""

    name: str
    description: str
    available_models: list[ModelInfo]
    connected: bool = False
    current_model: str | None = None


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str


@dataclass
class ToolCallRequest:
    """A tool call requested by the LLM."""
    
    id: str
    name: str
    arguments: dict


@dataclass
class GenerationResult:
    """Result from a generation request."""

    content: str
    model: str
    usage: dict | None = None  # Token usage info if available
    stop_reason: str | None = None
    tool_calls: list[ToolCallRequest] | None = None  # Tool calls if any


class BasePilot(ABC):
    """
    Abstract base class for LLM pilots.
    
    A pilot is the "brain" that can control the DAVM mech.
    Different pilots connect to different LLM backends (Anthropic, Ollama, etc.)
    """

    def __init__(self):
        self._connected = False
        self._current_model: str | None = None
        self._conversation_history: list[Message] = []
        self._system_prompt: str | None = None

    @property
    def connected(self) -> bool:
        """Check if the pilot is connected to its backend."""
        return self._connected

    @property
    def current_model(self) -> str | None:
        """Get the currently active model."""
        return self._current_model

    @property
    def conversation_history(self) -> list[Message]:
        """Get the current conversation history."""
        return self._conversation_history.copy()

    @abstractmethod
    async def connect(self, model: str | None = None) -> bool:
        """
        Connect to the LLM backend.
        
        Args:
            model: Optional model to use. If not specified, uses default.
            
        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the LLM backend."""
        pass

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """
        List available models for this pilot.
        
        Returns:
            List of available models with their info.
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user's message/prompt.
            system_prompt: Optional system prompt to set behavior.
            context: Optional conversation context (overrides internal history).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0 - 1.0).
            
        Returns:
            The generation result containing the response.
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: The user's message/prompt.
            system_prompt: Optional system prompt to set behavior.
            context: Optional conversation context (overrides internal history).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0 - 1.0).
            
        Yields:
            Chunks of the response as they're generated.
        """
        pass

    @abstractmethod
    def get_info(self) -> PilotInfo:
        """
        Get information about this pilot.
        
        Returns:
            PilotInfo with details about the pilot and its capabilities.
        """
        pass

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """
        Generate a response with tool/function calling support.
        
        Default implementation just calls generate() without tools.
        Subclasses should override to support native tool calling.
        
        Args:
            prompt: The user's message/prompt.
            tools: List of tool definitions (in provider-specific format).
            system_prompt: Optional system prompt.
            context: Optional conversation context.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            
        Returns:
            GenerationResult, potentially with tool_calls populated.
        """
        # Default: just generate without tools
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for conversations."""
        self._system_prompt = prompt

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self._conversation_history.append(Message(role=role, content=content))

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history = []

    def get_history_for_context(self) -> list[Message]:
        """Get conversation history formatted for context injection."""
        return self._conversation_history.copy()
