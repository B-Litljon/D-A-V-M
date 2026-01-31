# DAVM - Digital Assistant Virtual Mech

An LLM-powered agent framework that gives AI models a "body" to operate in the digital world.

## Concept

DAVM is like a "mech suit" for LLMs. Just as your brain pilots your body, an LLM (the "pilot") can pilot the DAVM mech, gaining access to:

- **Long-term Memory** - RAG database for persistent recall across conversations
- **File System** - Read, write, search, and organize files
- **Web Access** - Scrape websites and gather research
- **Autonomous Tool Use** - LLM can use these capabilities on its own

## Supported Pilots

- **Anthropic Claude** - claude-sonnet-4-20250514, claude-opus-4-20250514, etc.
- **Ollama** - Any local model (deepseek-r1, llama3, mistral, etc.)

## Quick Start

```bash
# Clone and enter the project
cd D-A-V-M

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Configure (copy and edit .env)
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY (optional if using Ollama)

# Run
davm
```

## CLI Commands

```
# Core commands:
/help              - Show all commands
/status            - Show mech status
/models            - List available models
/switch <pilot>    - Switch pilot (anthropic/ollama)
/model <name>      - Switch to specific model
/autonomy <level>  - Set autonomy (ask/semi/full)
/clear             - Clear conversation history

# Memory commands:
/memory            - Show memory statistics
/memory search <q> - Search through memories
/memory recent     - Show recent memories
/memory clear      - Clear all stored memories

# File commands:
/files             - Show file system status
/files list <path> - List directory contents
/files read <path> - Read a file
/files tree <path> - Show directory tree
/files search <p>  - Search for files

# Web commands:
/web               - Show web system status
/web fetch <url>   - Fetch and display a page
/web summary <url> - Get page summary

# Agent mode (autonomous tool use):
/agent <message>   - One-off agentic query
/agent on          - Enable agent mode for all messages
/agent off         - Disable agent mode

/exit              - Exit DAVM
```

## Autonomy Levels

Control how much freedom the LLM has to use tools:

- **ask** - Only memory tools available
- **semi** - Memory + file reading + web (no file writes)
- **full** - All tools including file write/delete

Set via `/autonomy <level>` or in `.env`:
```env
DAVM_AUTONOMY_LEVEL=semi
```

## Agent Mode

In agent mode, the LLM can autonomously use tools to help you:

```
You> /agent What Python files are in my project?

DAVM (Agent)>
Thinking with tools...
Used 1 tool(s)
  OK search_files

I found 15 Python files in your project. Here are the main ones:
- davm/__init__.py
- davm/core/mech.py
- davm/pilots/anthropic_pilot.py
...
```

## Systems

### Memory System (RAG)
- ChromaDB for vector storage
- all-MiniLM-L6-v2 embeddings
- Semantic search for relevant recall
- Persists across sessions

### File System
- Secure operations within allowed paths
- Read, write, search, delete
- Directory tree visualization

### Web System
- Fetch and parse web pages
- Extract text, links, metadata
- Async requests with httpx

### Tool System
- 13 tools across 4 categories
- Anthropic & Ollama compatible schemas
- Autonomy-aware permission checking

## Available Tools

| Tool | Category | Description |
|------|----------|-------------|
| search_memory | Memory | Search past conversations |
| get_recent_memories | Memory | Get recent memories |
| read_file | Files (Read) | Read file contents |
| list_directory | Files (Read) | List directory |
| search_files | Files (Read) | Find files by pattern |
| get_file_info | Files (Read) | Get file metadata |
| write_file | Files (Write) | Write to file |
| create_directory | Files (Write) | Create directory |
| delete_file | Files (Write) | Delete file |
| move_file | Files (Write) | Move/rename file |
| fetch_webpage | Web | Fetch and read page |
| get_webpage_summary | Web | Get page metadata |
| extract_links | Web | Extract page links |

## Configuration

```env
# API Keys
ANTHROPIC_API_KEY=your-key-here

# Defaults
DAVM_DEFAULT_PILOT=ollama
DAVM_OLLAMA_MODEL=deepseek-r1
DAVM_ANTHROPIC_MODEL=claude-sonnet-4-20250514

# File access
DAVM_ALLOWED_PATHS=~

# Autonomy level
DAVM_AUTONOMY_LEVEL=semi

# Storage
DAVM_DATA_DIR=./data
DAVM_MEMORY_DIR=./data/memory
```

## Instruction Booklet (Model Prompt)

See `DAVM_INSTRUCTION_BOOKLET.md` for the full model-facing prompt that explains how to operate the DAVM suit.

## Project Structure

```
davm/
  core/
    mech.py           - Main DAVM orchestrator
    config.py         - Settings management
  pilots/
    base.py           - Abstract pilot interface
    anthropic_pilot.py
    ollama_pilot.py
  systems/
    memory.py         - RAG with ChromaDB
    filesystem.py     - File operations
    web.py            - Web scraping
    tools.py          - Tool definitions & executor
  cli/
    app.py            - Interactive CLI
```

## Development Status

- [x] Phase 1: Core framework, pilots, CLI
- [x] Phase 2: RAG memory system
- [x] Phase 3: File system & web capabilities
- [x] Phase 4: Autonomous tool use

## Example Usage

```python
import asyncio
from davm import DAVM

async def main():
    mech = DAVM()
    await mech.activate(pilot="ollama", model="llama3")
    
    # Simple chat (with memory)
    result = await mech.chat("My name is Alex")
    print(result.content)
    
    # Agentic chat (with tools)
    response, tools_used = await mech.chat_with_tools(
        "What files are in my home directory?"
    )
    print(response)
    print(f"Used {len(tools_used)} tools")
    
    # Direct system access
    files = mech.list_files("~/Documents")
    page = await mech.fetch_url("https://example.com")
    memories = await mech.search_memory("favorite color")
    
    await mech.deactivate()

asyncio.run(main())
```

---

**DAVM** - Your LLM's body in the digital realm.
