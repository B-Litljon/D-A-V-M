"""
DAVM Memory System - RAG-based long-term memory using ChromaDB.

This system gives the mech persistent memory across conversations.
It stores conversation exchanges as embeddings and retrieves relevant
past context when generating new responses.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class Memory:
    """A single memory unit (typically a conversation exchange)."""
    
    id: str
    content: str  # The actual text content
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Optional fields for richer context
    user_message: str | None = None
    assistant_response: str | None = None
    conversation_id: str | None = None
    summary: str | None = None
    importance: float = 0.5  # 0.0 to 1.0, for prioritizing memories
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "importance": self.importance,
            **self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create Memory from dictionary."""
        # Extract known fields
        known_fields = {
            "id", "content", "timestamp", "user_message", 
            "assistant_response", "conversation_id", "summary", "importance"
        }
        metadata = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            id=data["id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_message=data.get("user_message"),
            assistant_response=data.get("assistant_response"),
            conversation_id=data.get("conversation_id"),
            summary=data.get("summary"),
            importance=data.get("importance", 0.5),
            metadata=metadata,
        )


@dataclass
class MemorySearchResult:
    """Result from a memory search."""
    
    memory: Memory
    relevance_score: float  # Higher = more relevant (0.0 to 1.0)
    distance: float  # Raw distance from ChromaDB


class MemorySystem:
    """
    RAG-based memory system for DAVM.
    
    Uses ChromaDB to store conversation memories as vector embeddings,
    enabling semantic search to retrieve relevant past context.
    
    How it works:
    1. After each conversation exchange, the content is embedded and stored
    2. Before generating a response, relevant past memories are retrieved
    3. These memories are injected into the context to help the LLM "remember"
    
    Example:
        memory = MemorySystem(persist_dir="./data/memory")
        
        # Store a memory
        memory.store(
            user_message="What's my favorite color?",
            assistant_response="You mentioned your favorite color is blue."
        )
        
        # Later, retrieve relevant memories
        results = memory.search("Tell me about colors I like", limit=5)
        for result in results:
            print(f"Relevance: {result.relevance_score:.2f}")
            print(f"Content: {result.memory.content}")
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "davm_memories",
    ):
        """
        Initialize the memory system.
        
        Args:
            persist_dir: Directory for persistent storage. If None, uses in-memory.
            collection_name: Name of the ChromaDB collection.
        """
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._current_conversation_id: str | None = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Check if the memory system is initialized."""
        return self._initialized

    @property
    def current_conversation_id(self) -> str | None:
        """Get the current conversation ID."""
        return self._current_conversation_id

    async def initialize(self) -> None:
        """Initialize the ChromaDB client and collection."""
        if self._initialized:
            return

        # Create persist directory if needed
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Use persistent client
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        else:
            # Use ephemeral client for testing
            self._client = chromadb.EphemeralClient()

        # Get or create the collection
        # ChromaDB will use its default embedding function (all-MiniLM-L6-v2)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "DAVM long-term memory storage"},
        )
        
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the memory system."""
        # ChromaDB PersistentClient auto-persists, no explicit save needed
        self._client = None
        self._collection = None
        self._initialized = False

    def start_conversation(self, conversation_id: str | None = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            conversation_id: Optional ID for the conversation. 
                           If None, generates a new UUID.
        
        Returns:
            The conversation ID.
        """
        self._current_conversation_id = conversation_id or str(uuid.uuid4())
        return self._current_conversation_id

    def end_conversation(self) -> None:
        """End the current conversation session."""
        self._current_conversation_id = None

    async def store(
        self,
        user_message: str,
        assistant_response: str,
        summary: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """
        Store a conversation exchange as a memory.
        
        Args:
            user_message: The user's message.
            assistant_response: The assistant's response.
            summary: Optional summary of the exchange.
            importance: Importance score (0.0 to 1.0).
            metadata: Additional metadata to store.
        
        Returns:
            The created Memory object.
        """
        if not self._initialized or not self._collection:
            raise RuntimeError("Memory system not initialized. Call initialize() first.")

        # Create the content to embed (combines both for better semantic matching)
        content = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Generate a deterministic ID based on content and timestamp
        timestamp = datetime.now()
        id_source = f"{content}{timestamp.isoformat()}"
        memory_id = hashlib.sha256(id_source.encode()).hexdigest()[:16]
        
        # Create the memory object
        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            user_message=user_message,
            assistant_response=assistant_response,
            conversation_id=self._current_conversation_id,
            summary=summary,
            importance=importance,
            metadata=metadata or {},
        )
        
        # Prepare metadata for ChromaDB (must be flat dict with simple types)
        chroma_metadata = {
            "timestamp": timestamp.isoformat(),
            "importance": importance,
            "user_message_preview": user_message[:200] if user_message else "",
            "assistant_response_preview": assistant_response[:200] if assistant_response else "",
        }
        
        if self._current_conversation_id:
            chroma_metadata["conversation_id"] = self._current_conversation_id
        if summary:
            chroma_metadata["summary"] = summary
        if metadata:
            # Add any simple metadata (strings, numbers, bools only)
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    chroma_metadata[k] = v

        # Store in ChromaDB
        self._collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[chroma_metadata],
        )
        
        return memory

    async def search(
        self,
        query: str,
        limit: int = 5,
        min_relevance: float = 0.0,
        conversation_id: str | None = None,
        include_current_conversation: bool = True,
    ) -> list[MemorySearchResult]:
        """
        Search for relevant memories.
        
        Args:
            query: The search query (will be embedded for semantic search).
            limit: Maximum number of results to return.
            min_relevance: Minimum relevance score (0.0 to 1.0).
            conversation_id: Filter to specific conversation.
            include_current_conversation: Whether to include memories from current conversation.
        
        Returns:
            List of MemorySearchResult objects, sorted by relevance.
        """
        if not self._initialized or not self._collection:
            raise RuntimeError("Memory system not initialized. Call initialize() first.")

        # Build where filter
        where_filter = None
        if conversation_id:
            where_filter = {"conversation_id": conversation_id}
        elif not include_current_conversation and self._current_conversation_id:
            where_filter = {"conversation_id": {"$ne": self._current_conversation_id}}

        # Query ChromaDB
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # Handle case where collection is empty or other errors
            return []

        # Process results
        search_results = []
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, memory_id in enumerate(ids):
            # Convert distance to relevance score (ChromaDB uses L2 distance by default)
            # Lower distance = higher relevance
            distance = distances[i] if i < len(distances) else 1.0
            # Convert to 0-1 scale (approximate, depends on embedding space)
            relevance = max(0.0, 1.0 - (distance / 2.0))
            
            if relevance < min_relevance:
                continue

            # Reconstruct memory from stored data
            metadata = metadatas[i] if i < len(metadatas) else {}
            content = documents[i] if i < len(documents) else ""
            
            memory = Memory(
                id=memory_id,
                content=content,
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                user_message=metadata.get("user_message_preview"),
                assistant_response=metadata.get("assistant_response_preview"),
                conversation_id=metadata.get("conversation_id"),
                summary=metadata.get("summary"),
                importance=metadata.get("importance", 0.5),
            )
            
            search_results.append(MemorySearchResult(
                memory=memory,
                relevance_score=relevance,
                distance=distance,
            ))

        # Sort by relevance (highest first)
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return search_results

    async def get_context_for_prompt(
        self,
        current_prompt: str,
        max_memories: int = 5,
        max_tokens: int = 2000,
    ) -> str:
        """
        Get formatted context from relevant memories for injection into a prompt.
        
        This is the main method used by DAVM to augment prompts with
        relevant past memories.
        
        Args:
            current_prompt: The user's current prompt/message.
            max_memories: Maximum number of memories to include.
            max_tokens: Approximate max tokens for the context (rough estimate).
        
        Returns:
            Formatted string of relevant memories, or empty string if none found.
        """
        results = await self.search(
            query=current_prompt,
            limit=max_memories,
            min_relevance=0.3,  # Only include reasonably relevant memories
            include_current_conversation=False,  # Don't repeat current conversation
        )
        
        if not results:
            return ""

        # Build context string
        context_parts = ["[Relevant memories from past conversations:]"]
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate: ~4 chars per token
        
        for result in results:
            memory = result.memory
            
            # Format the memory
            memory_text = f"\n---\n"
            if memory.summary:
                memory_text += f"Summary: {memory.summary}\n"
            memory_text += f"Time: {memory.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            memory_text += f"{memory.content}\n"
            memory_text += f"(Relevance: {result.relevance_score:.0%})"
            
            # Check if we have room
            if total_chars + len(memory_text) > char_limit:
                break
                
            context_parts.append(memory_text)
            total_chars += len(memory_text)
        
        if len(context_parts) == 1:
            return ""  # No memories added
            
        context_parts.append("\n[End of memories]\n")
        return "\n".join(context_parts)

    async def get_memory_count(self) -> int:
        """Get the total number of stored memories."""
        if not self._initialized or not self._collection:
            return 0
        return self._collection.count()

    async def get_recent_memories(
        self,
        limit: int = 10,
        conversation_id: str | None = None,
    ) -> list[Memory]:
        """
        Get the most recent memories.
        
        Args:
            limit: Maximum number of memories to return.
            conversation_id: Filter to specific conversation.
        
        Returns:
            List of Memory objects, most recent first.
        """
        if not self._initialized or not self._collection:
            return []

        # ChromaDB doesn't support sorting by metadata directly,
        # so we fetch all and sort in Python (not ideal for large datasets)
        where_filter = None
        if conversation_id:
            where_filter = {"conversation_id": conversation_id}

        try:
            results = self._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        if not results or not results.get("ids"):
            return []

        memories = []
        for i, memory_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results.get("metadatas") else {}
            content = results["documents"][i] if results.get("documents") else ""
            
            memories.append(Memory(
                id=memory_id,
                content=content,
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                user_message=metadata.get("user_message_preview"),
                assistant_response=metadata.get("assistant_response_preview"),
                conversation_id=metadata.get("conversation_id"),
                summary=metadata.get("summary"),
                importance=metadata.get("importance", 0.5),
            ))

        # Sort by timestamp (most recent first)
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        
        return memories[:limit]

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.
        
        Args:
            memory_id: The ID of the memory to delete.
        
        Returns:
            True if deleted, False if not found.
        """
        if not self._initialized or not self._collection:
            return False

        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    async def clear_all(self) -> int:
        """
        Clear all memories.
        
        Returns:
            Number of memories deleted.
        """
        if not self._initialized or not self._collection:
            return 0

        count = self._collection.count()
        
        # Delete and recreate collection
        if self._client:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"description": "DAVM long-term memory storage"},
            )
        
        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        if not self._initialized or not self._collection:
            return {"initialized": False}

        count = self._collection.count()
        
        return {
            "initialized": True,
            "total_memories": count,
            "collection_name": self._collection_name,
            "persist_dir": str(self._persist_dir) if self._persist_dir else "in-memory",
            "current_conversation_id": self._current_conversation_id,
        }
