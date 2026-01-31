"""Ollama local model pilot implementation."""

import re
from typing import Any, AsyncGenerator

import ollama
from ollama import AsyncClient

from davm.core.config import get_settings
from davm.pilots.base import (
    BasePilot,
    GenerationResult,
    Message,
    ModelInfo,
    PilotInfo,
    ToolCallRequest,
)


class OllamaPilot(BasePilot):
    """
    Pilot implementation for Ollama local models.
    
    Connects to a local Ollama instance to use open-source models
    as the brain piloting the DAVM mech.
    """

    def __init__(self):
        super().__init__()
        self._client: AsyncClient | None = None
        self._available_models: list[ModelInfo] = []
        self._host: str = ""

    def _clean_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks from reasoning model output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def connect(self, model: str | None = None) -> bool:
        """Connect to the Ollama server and select a model."""
        settings = get_settings()
        self._host = settings.ollama_host

        try:
            self._client = AsyncClient(host=self._host)
            
            # Fetch available models
            await self._refresh_models()
            
            if not self._available_models:
                raise ConnectionError(
                    "No models available in Ollama. "
                    "Pull a model first: ollama pull <model-name>"
                )

            # Use provided model or default
            target_model = model or settings.davm_ollama_model
            
            # Check if target model is available
            available_names = [m.name for m in self._available_models]
            
            if target_model not in available_names:
                # Try to find a partial match (e.g., "deepseek-r1" matches "deepseek-r1:latest")
                matched = None
                for name in available_names:
                    if target_model in name or name.startswith(target_model):
                        matched = name
                        break
                
                if matched:
                    target_model = matched
                else:
                    # Fall back to first available model
                    target_model = available_names[0]
                    
            self._current_model = target_model
            self._connected = True
            return True

        except ollama.ResponseError as e:
            self._connected = False
            raise ConnectionError(f"Ollama error: {e}")
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Ollama at {self._host}: {e}")

    async def _refresh_models(self) -> None:
        """Refresh the list of available models from Ollama."""
        if not self._client:
            return

        try:
            response = await self._client.list()
            self._available_models = []
            
            # Response is a ListResponse object with a 'models' attribute
            models = response.models if hasattr(response, 'models') else []
            
            for model in models:
                # Model is an object with attributes, not a dict
                name = model.model if hasattr(model, 'model') else str(model)
                size_bytes = model.size if hasattr(model, 'size') else 0
                
                # Convert size to human-readable format
                if size_bytes:
                    size_gb = size_bytes / (1024 ** 3)
                    size_str = f"{size_gb:.1f} GB"
                else:
                    size_str = "Unknown"
                
                # Get parameter size from details if available
                description = f"Local model: {name}"
                if hasattr(model, 'details') and model.details:
                    if hasattr(model.details, 'parameter_size') and model.details.parameter_size:
                        description = f"{model.details.parameter_size} parameters"

                self._available_models.append(
                    ModelInfo(
                        name=name,
                        size=size_str,
                        description=description,
                        capabilities=["chat", "completion"],
                    )
                )

        except Exception as e:
            # Keep existing models list if refresh fails
            pass

    async def disconnect(self) -> None:
        """Disconnect from Ollama."""
        self._client = None
        self._connected = False
        self._current_model = None

    async def list_models(self) -> list[ModelInfo]:
        """List available Ollama models."""
        if self._client:
            await self._refresh_models()
        return self._available_models.copy()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate a response from an Ollama model."""
        if not self._connected or not self._client:
            raise RuntimeError("Pilot not connected. Call connect() first.")

        # Build messages list
        messages = []
        
        # Add system prompt first
        sys_prompt = system_prompt or self._system_prompt or (
            "You are a helpful AI assistant operating within the DAVM "
            "(Digital Assistant Virtual Mech) framework. You have access to "
            "various capabilities including memory, file system access, and "
            "web browsing. Be helpful, accurate, and respect the user's "
            "autonomy preferences."
        )
        messages.append({"role": "system", "content": sys_prompt})

        # Add context/history
        history = context if context is not None else self._conversation_history
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add the current prompt
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat(
                model=self._current_model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )

            raw_content = response.get("message", {}).get("content", "")
            # Clean thinking tags from reasoning models
            content = self._clean_thinking(raw_content)
            
            # Update conversation history with cleaned content
            self.add_message("user", prompt)
            self.add_message("assistant", content)

            # Extract usage info if available
            usage = None
            if "eval_count" in response or "prompt_eval_count" in response:
                usage = {
                    "input_tokens": response.get("prompt_eval_count", 0),
                    "output_tokens": response.get("eval_count", 0),
                }

            return GenerationResult(
                content=content,
                model=self._current_model,
                usage=usage,
                stop_reason=response.get("done_reason"),
            )

        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama error: {e}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: list[Message] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from an Ollama model."""
        if not self._connected or not self._client:
            raise RuntimeError("Pilot not connected. Call connect() first.")

        # Build messages list
        messages = []
        
        sys_prompt = system_prompt or self._system_prompt or (
            "You are a helpful AI assistant operating within the DAVM "
            "(Digital Assistant Virtual Mech) framework."
        )
        messages.append({"role": "system", "content": sys_prompt})

        history = context if context is not None else self._conversation_history
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        try:
            # State machine for filtering <think>...</think> blocks
            inside_thought = False
            buffer = ""
            full_response = ""

            async for chunk in await self._client.chat(
                model=self._current_model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                stream=True,
            ):
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue

                buffer += content
                
                # Check entry/exit of thought mode
                if "<think>" in buffer and not inside_thought:
                    inside_thought = True
                
                if "</think>" in buffer and inside_thought:
                    # Clear thought from buffer, keep the rest
                    _, remaining = buffer.split("</think>", 1)
                    buffer = remaining
                    inside_thought = False
                
                # Yield only if we are not thinking and not potentially starting a tag
                if not inside_thought:
                    if "<" not in buffer:
                        yield buffer
                        full_response += buffer
                        buffer = ""
            
            # Flush remaining buffer
            if buffer and not inside_thought:
                yield buffer
                full_response += buffer

            # Update history with clean output
            clean_full = self._clean_thinking(full_response)
            self.add_message("user", prompt)
            self.add_message("assistant", clean_full)

        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama error: {e}")

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
        
        Ollama supports OpenAI-compatible tool calling for models that
        support it (like llama3.2, mistral, etc.). For models that don't,
        this will just return a regular generation.
        """
        if not self._connected or not self._client:
            raise RuntimeError("Pilot not connected. Call connect() first.")

        # Build messages list
        messages = []
        
        sys_prompt = system_prompt or self._system_prompt or (
            "You are a helpful AI assistant operating within the DAVM "
            "(Digital Assistant Virtual Mech) framework. You have access to "
            "tools for memory, file system, and web browsing. Use them when "
            "appropriate to help the user. Be helpful and accurate."
        )
        messages.append({"role": "system", "content": sys_prompt})

        history = context if context is not None else self._conversation_history
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        # Only add prompt if it's not empty (subsequent rounds may have empty prompts)
        if prompt:
            messages.append({"role": "user", "content": prompt})

        try:
            # Try with tools - Ollama will ignore them if model doesn't support it
            response = await self._client.chat(
                model=self._current_model,
                messages=messages,
                tools=tools if tools else None,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )

            # Extract content and tool calls
            message = response.get("message", {})
            content = message.get("content", "")
            
            # Check for tool calls in the response
            tool_calls = []
            raw_tool_calls = message.get("tool_calls") or []
            
            for i, tc in enumerate(raw_tool_calls):
                # Ollama format: {"function": {"name": "...", "arguments": {...}}}
                func = tc.get("function", {})
                tool_calls.append(ToolCallRequest(
                    id=f"call_{i}_{func.get('name', 'unknown')}",
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))

            # Update conversation history
            if prompt:
                self.add_message("user", prompt)
            if content:
                self.add_message("assistant", content)

            # Extract usage info if available
            usage = None
            if "eval_count" in response or "prompt_eval_count" in response:
                usage = {
                    "input_tokens": response.get("prompt_eval_count", 0),
                    "output_tokens": response.get("eval_count", 0),
                }

            return GenerationResult(
                content=content,
                model=self._current_model,
                usage=usage,
                stop_reason=response.get("done_reason"),
                tool_calls=tool_calls if tool_calls else None,
            )

        except ollama.ResponseError as e:
            # If tool calling fails, fall back to regular generation
            if "tool" in str(e).lower():
                return await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    context=context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            raise RuntimeError(f"Ollama error: {e}")

    def get_info(self) -> PilotInfo:
        """Get information about the Ollama pilot."""
        return PilotInfo(
            name="Ollama (Local)",
            description=f"Local LLM models via Ollama at {self._host}",
            available_models=self._available_models,
            connected=self._connected,
            current_model=self._current_model,
        )
