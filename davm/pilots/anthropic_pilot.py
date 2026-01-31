"""Anthropic Claude pilot implementation."""

import json
from typing import Any, AsyncGenerator

import anthropic

from davm.core.config import get_settings
from davm.pilots.base import (
    BasePilot,
    GenerationResult,
    Message,
    ModelInfo,
    PilotInfo,
    ToolCallRequest,
)


# Available Claude models with their info
CLAUDE_MODELS = {
    "claude-opus-4-20250514": ModelInfo(
        name="claude-opus-4-20250514",
        description="Most capable Claude model for complex tasks",
        context_window=200000,
        capabilities=["analysis", "coding", "writing", "reasoning", "vision"],
    ),
    "claude-sonnet-4-20250514": ModelInfo(
        name="claude-sonnet-4-20250514",
        description="Balanced performance and speed",
        context_window=200000,
        capabilities=["analysis", "coding", "writing", "reasoning", "vision"],
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        name="claude-3-5-haiku-20241022",
        description="Fast and efficient for simpler tasks",
        context_window=200000,
        capabilities=["analysis", "coding", "writing", "reasoning"],
    ),
}


class AnthropicPilot(BasePilot):
    """
    Pilot implementation for Anthropic's Claude models.
    
    Connects to the Anthropic API to use Claude as the brain
    piloting the DAVM mech.
    """

    def __init__(self):
        super().__init__()
        self._client: anthropic.AsyncAnthropic | None = None
        self._available_models: list[ModelInfo] = list(CLAUDE_MODELS.values())

    async def connect(self, model: str | None = None) -> bool:
        """Connect to the Anthropic API."""
        settings = get_settings()
        
        if not settings.anthropic_api_key:
            raise ValueError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY in your .env file."
            )

        try:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key
            )
            
            # Use provided model or default
            target_model = model or settings.davm_anthropic_model
            
            # Validate model exists
            if target_model not in CLAUDE_MODELS:
                # Allow custom model names (for newer models not in our list)
                self._available_models.append(
                    ModelInfo(name=target_model, description="Custom model")
                )
            
            self._current_model = target_model
            self._connected = True
            return True
            
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Anthropic: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the Anthropic API."""
        if self._client:
            await self._client.close()
        self._client = None
        self._connected = False
        self._current_model = None

    async def list_models(self) -> list[ModelInfo]:
        """List available Claude models."""
        return self._available_models.copy()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate a response from Claude."""
        if not self._connected or not self._client:
            raise RuntimeError("Pilot not connected. Call connect() first.")

        # Build messages list
        messages = []
        
        # Add context/history
        history = context if context is not None else self._conversation_history
        for msg in history:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

        # Add the current prompt
        messages.append({"role": "user", "content": prompt})

        # Use provided system prompt, or instance default, or a basic default
        sys_prompt = system_prompt or self._system_prompt or (
            "You are a helpful AI assistant operating within the DAVM "
            "(Digital Assistant Virtual Mech) framework. You have access to "
            "various capabilities including memory, file system access, and "
            "web browsing. Be helpful, accurate, and respect the user's "
            "autonomy preferences."
        )

        try:
            response = await self._client.messages.create(
                model=self._current_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_prompt,
                messages=messages,
            )

            content = response.content[0].text if response.content else ""
            
            # Update conversation history
            self.add_message("user", prompt)
            self.add_message("assistant", content)

            return GenerationResult(
                content=content,
                model=self._current_model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                stop_reason=response.stop_reason,
            )

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Claude."""
        if not self._connected or not self._client:
            raise RuntimeError("Pilot not connected. Call connect() first.")

        # Build messages list
        messages = []
        
        history = context if context is not None else self._conversation_history
        for msg in history:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        sys_prompt = system_prompt or self._system_prompt or (
            "You are a helpful AI assistant operating within the DAVM "
            "(Digital Assistant Virtual Mech) framework."
        )

        try:
            full_response = ""
            
            async with self._client.messages.stream(
                model=self._current_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    yield text

            # Update conversation history after streaming completes
            self.add_message("user", prompt)
            self.add_message("assistant", full_response)

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate a response with tool calling support."""
        if not self._connected or not self._client:
            raise RuntimeError("Pilot not connected. Call connect() first.")

        # Build messages list
        messages = []
        
        history = context if context is not None else self._conversation_history
        for msg in history:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        sys_prompt = system_prompt or self._system_prompt or (
            "You are a helpful AI assistant operating within the DAVM "
            "(Digital Assistant Virtual Mech) framework. You have access to "
            "tools for memory, file system, and web browsing. Use them when "
            "appropriate to help the user. Be helpful and accurate."
        )

        try:
            response = await self._client.messages.create(
                model=self._current_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_prompt,
                messages=messages,
                tools=tools,
            )

            # Process response - may contain text and/or tool calls
            content = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCallRequest(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    ))

            # Update conversation history
            self.add_message("user", prompt)
            if content:
                self.add_message("assistant", content)

            return GenerationResult(
                content=content,
                model=self._current_model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                stop_reason=response.stop_reason,
                tool_calls=tool_calls if tool_calls else None,
            )

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def get_info(self) -> PilotInfo:
        """Get information about the Anthropic pilot."""
        return PilotInfo(
            name="Anthropic Claude",
            description="Anthropic's Claude family of AI assistants",
            available_models=self._available_models,
            connected=self._connected,
            current_model=self._current_model,
        )
