"""
DAVM CLI - Command-line interface for the Digital Assistant Virtual Mech.
"""

import asyncio
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from davm.core.mech import DAVM, PilotType


# Rich console for pretty output
console = Console()

# Stream markers for thinking output
THINK_START_TOKEN = "[[DAVM_THINK_START]]"
THINK_END_TOKEN = "[[DAVM_THINK_END]]"
THINK_CHUNK_TOKEN = "[[DAVM_THINK_CHUNK]]"

# Custom prompt style
prompt_style = Style.from_dict({
    "prompt": "ansicyan bold",
})


def print_banner():
    """Print the DAVM welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ██████╗  █████╗ ██╗   ██╗███╗   ███╗                    ║
    ║     ██╔══██╗██╔══██╗██║   ██║████╗ ████║                    ║
    ║     ██║  ██║███████║██║   ██║██╔████╔██║                    ║
    ║     ██║  ██║██╔══██║╚██╗ ██╔╝██║╚██╔╝██║                    ║
    ║     ██████╔╝██║  ██║ ╚████╔╝ ██║ ╚═╝ ██║                    ║
    ║     ╚═════╝ ╚═╝  ╚═╝  ╚═══╝  ╚═╝     ╚═╝                    ║
    ║                                                              ║
    ║     Digital Assistant Virtual Mech                           ║
    ║     Your LLM-powered agent with persistent memory            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="cyan")


def load_instruction_booklet() -> str | None:
    """Load the DAVM instruction booklet for the system prompt."""
    booklet_path = Path(__file__).resolve().parents[2] / "DAVM_INSTRUCTION_BOOKLET.md"
    try:
        return booklet_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None


def apply_instruction_booklet(mech: DAVM) -> None:
    """Apply the instruction booklet to the mech's system prompt."""
    booklet = load_instruction_booklet()
    if booklet:
        mech.set_system_prompt(booklet)
        console.print("[dim]Loaded DAVM instruction booklet.[/dim]")


def print_help():
    """Print available commands."""
    table = Table(title="DAVM Commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    
    commands = [
        ("/help", "Show this help message"),
        ("/status", "Show mech status including memory"),
        ("/models", "List available models"),
        ("/switch <pilot>", "Switch pilot (anthropic/ollama)"),
        ("/model <name>", "Switch to specific model"),
        ("/autonomy <level>", "Set autonomy level (ask/semi/full)"),
        ("/thinking <on|off>", "Toggle thinking output in CLI"),
        ("/clear", "Clear conversation history"),
        ("/history", "Show conversation history"),
        ("", ""),
        ("[bold]Memory Commands[/bold]", ""),
        ("/memory", "Show memory statistics"),
        ("/memory search <query>", "Search through memories"),
        ("/memory recent", "Show recent memories"),
        ("/memory clear", "Clear all stored memories"),
        ("/memory off", "Disable memory for this session"),
        ("/memory on", "Re-enable memory"),
        ("", ""),
        ("[bold]File Commands[/bold]", ""),
        ("/files", "Show file system status"),
        ("/files list <path>", "List directory contents"),
        ("/files read <path>", "Read a file"),
        ("/files tree <path>", "Show directory tree"),
        ("/files search <pattern>", "Search for files"),
        ("", ""),
        ("[bold]Web Commands[/bold]", ""),
        ("/web", "Show web system status"),
        ("/web fetch <url>", "Fetch and display a web page"),
        ("/web summary <url>", "Get a summary of a web page"),
        ("/web links <url>", "List links from a web page"),
        ("", ""),
        ("[bold]Agent Mode[/bold]", ""),
        ("/agent <message>", "Let the LLM use tools autonomously"),
        ("/agent on", "Enable agent mode for all messages"),
        ("/agent off", "Disable agent mode (default)"),
        ("", ""),
        ("/exit, /quit", "Exit DAVM"),
    ]
    
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    
    console.print(table)


async def print_status(mech: DAVM):
    """Print current mech status."""
    status = await mech.get_status()
    
    table = Table(title="DAVM Status", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    
    table.add_row("Active", "[green]Yes[/green]" if status.active else "[red]No[/red]")
    table.add_row("Pilot", status.pilot_type or "None")
    
    if status.pilot_info:
        table.add_row("Model", status.pilot_info.current_model or "None")
    
    table.add_row("Autonomy Level", status.autonomy_level)
    table.add_row("Active Systems", ", ".join(status.systems_active) or "None")
    
    # Memory stats
    if status.memory_stats:
        table.add_row("", "")  # Separator
        table.add_row("[bold]Memory[/bold]", "")
        table.add_row("  Status", "[green]Enabled[/green]" if status.memory_stats.get("initialized") else "[yellow]Disabled[/yellow]")
        table.add_row("  Total Memories", str(status.memory_stats.get("total_memories", 0)))
        table.add_row("  Storage", status.memory_stats.get("persist_dir", "in-memory"))
    
    console.print(table)


async def print_models(mech: DAVM, pilot: PilotType | None = None):
    """Print available models."""
    try:
        models = await mech.list_available_models(pilot=pilot)
        
        title = f"Available Models"
        if pilot:
            title += f" ({pilot})"
        elif mech.pilot_type:
            title += f" ({mech.pilot_type})"
            
        table = Table(title=title, show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Size")
        table.add_column("Description")
        
        for model in models:
            current = ""
            if mech.pilot and mech.pilot.current_model == model.name:
                current = " [green](active)[/green]"
            table.add_row(
                model.name + current,
                model.size or "-",
                model.description or "-"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")


def print_history(mech: DAVM):
    """Print conversation history."""
    history = mech.get_history()
    
    if not history:
        console.print("[dim]No conversation history.[/dim]")
        return
    
    console.print(Panel("Conversation History", style="cyan"))
    
    for msg in history:
        if msg.role == "user":
            console.print(f"[cyan bold]You:[/cyan bold] {msg.content}")
        else:
            console.print(f"[green bold]DAVM:[/green bold] {msg.content[:200]}...")
        console.print()


async def handle_memory_command(args: str | None, mech: DAVM) -> None:
    """Handle memory-related commands."""
    if not args:
        # Show memory stats
        stats = await mech.get_memory_stats()
        
        if not stats.get("enabled"):
            console.print("[yellow]Memory system is disabled.[/yellow]")
            console.print("Use [cyan]/memory on[/cyan] to enable it.")
            return
        
        table = Table(title="Memory Statistics", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        
        table.add_row("Status", "[green]Enabled[/green]" if stats.get("initialized") else "[red]Not initialized[/red]")
        table.add_row("Total Memories", str(stats.get("total_memories", 0)))
        table.add_row("Storage Location", stats.get("persist_dir", "in-memory"))
        if stats.get("current_conversation_id"):
            table.add_row("Current Conversation", stats["current_conversation_id"][:8] + "...")
        
        console.print(table)
        return
    
    parts = args.split(maxsplit=1)
    subcmd = parts[0].lower()
    subarg = parts[1] if len(parts) > 1 else None
    
    if subcmd == "search":
        if not subarg:
            console.print("[red]Usage: /memory search <query>[/red]")
            return
        
        if not mech.memory_enabled:
            console.print("[yellow]Memory is disabled. Enable with /memory on[/yellow]")
            return
        
        console.print(f"[dim]Searching memories for: {subarg}[/dim]")
        results = await mech.search_memory(subarg, limit=5)
        
        if not results:
            console.print("[dim]No relevant memories found.[/dim]")
            return
        
        console.print(f"\n[bold]Found {len(results)} relevant memories:[/bold]\n")
        
        for i, result in enumerate(results, 1):
            relevance_color = "green" if result.relevance_score > 0.7 else "yellow" if result.relevance_score > 0.4 else "dim"
            console.print(f"[{relevance_color}]#{i} Relevance: {result.relevance_score:.0%}[/{relevance_color}]")
            console.print(f"[dim]{result.memory.timestamp.strftime('%Y-%m-%d %H:%M')}[/dim]")
            
            # Show the memory content (truncated)
            content = result.memory.content
            if len(content) > 300:
                content = content[:300] + "..."
            console.print(Panel(content, border_style="dim"))
            console.print()
    
    elif subcmd == "recent":
        if not mech.memory_enabled or not mech.memory:
            console.print("[yellow]Memory is disabled. Enable with /memory on[/yellow]")
            return
        
        memories = await mech.memory.get_recent_memories(limit=5)
        
        if not memories:
            console.print("[dim]No memories stored yet.[/dim]")
            return
        
        console.print(f"\n[bold]Recent memories ({len(memories)}):[/bold]\n")
        
        for i, mem in enumerate(memories, 1):
            console.print(f"[cyan]#{i}[/cyan] [dim]{mem.timestamp.strftime('%Y-%m-%d %H:%M')}[/dim]")
            content = mem.content
            if len(content) > 200:
                content = content[:200] + "..."
            console.print(Panel(content, border_style="dim"))
    
    elif subcmd == "clear":
        if not mech.memory_enabled:
            console.print("[yellow]Memory is already disabled.[/yellow]")
            return
        
        console.print("[yellow]Are you sure you want to clear ALL memories? This cannot be undone.[/yellow]")
        console.print("Type 'yes' to confirm:")
        
        # Simple confirmation (in a real app, use prompt_toolkit)
        confirm = input().strip().lower()
        if confirm == "yes":
            count = await mech.clear_memory()
            console.print(f"[green]Cleared {count} memories.[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")
    
    elif subcmd == "off":
        await mech.disable_memory()
        console.print("[yellow]Memory system disabled for this session.[/yellow]")
        console.print("Conversations will not be stored or recalled.")
    
    elif subcmd == "on":
        if mech.memory_enabled:
            console.print("[green]Memory is already enabled.[/green]")
        else:
            await mech.enable_memory()
            console.print("[green]Memory system enabled.[/green]")
            console.print("Conversations will now be stored and past context will be recalled.")
    
    else:
        console.print(f"[red]Unknown memory command: {subcmd}[/red]")
        console.print("Available: search, recent, clear, on, off")


async def handle_files_command(args: str | None, mech: DAVM) -> None:
    """Handle file system commands."""
    if not args:
        # Show file system status
        if not mech.files_enabled:
            console.print("[yellow]File system is disabled.[/yellow]")
            return
        
        fs = mech.files
        if fs:
            table = Table(title="File System Status", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value")
            table.add_row("Status", "[green]Enabled[/green]")
            table.add_row("Allowed Paths", "\n".join(str(p) for p in fs.allowed_paths))
            console.print(table)
        return
    
    parts = args.split(maxsplit=1)
    subcmd = parts[0].lower()
    subarg = parts[1] if len(parts) > 1 else None
    
    if subcmd == "list":
        if not subarg:
            subarg = "~"  # Default to home
        
        if not mech.files_enabled:
            console.print("[yellow]File system is disabled.[/yellow]")
            return
        
        files = mech.list_files(subarg)
        if not files:
            console.print(f"[dim]No files found or access denied: {subarg}[/dim]")
            return
        
        table = Table(title=f"Files in {subarg}", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Size")
        table.add_column("Modified")
        
        for f in files[:50]:  # Limit to 50
            file_type = "[blue]DIR[/blue]" if f.is_dir else f.extension or "file"
            size = "-" if f.is_dir else f.size_human
            modified = f.modified.strftime("%Y-%m-%d %H:%M")
            table.add_row(f.name, file_type, size, modified)
        
        if len(files) > 50:
            table.add_row(f"... and {len(files) - 50} more", "", "", "")
        
        console.print(table)
    
    elif subcmd == "read":
        if not subarg:
            console.print("[red]Usage: /files read <path>[/red]")
            return
        
        if not mech.files_enabled:
            console.print("[yellow]File system is disabled.[/yellow]")
            return
        
        result = mech.read_file(subarg)
        if result.success:
            content = result.data if result.data else ""
            if isinstance(content, bytes):
                content = f"[Binary file: {len(content)} bytes]"
            elif len(content) > 2000:
                content = content[:2000] + "\n... [truncated]"
            console.print(Panel(content, title=subarg, border_style="green"))
        else:
            console.print(f"[red]{result.message}[/red]")
    
    elif subcmd == "tree":
        if not subarg:
            subarg = "~"
        
        if not mech.files_enabled:
            console.print("[yellow]File system is disabled.[/yellow]")
            return
        
        tree = mech.get_file_tree(subarg, max_depth=3)
        console.print(Panel(tree, title="Directory Tree", border_style="cyan"))
    
    elif subcmd == "search":
        if not subarg:
            console.print("[red]Usage: /files search <pattern>[/red]")
            console.print("Example: /files search *.py")
            return
        
        if not mech.files_enabled:
            console.print("[yellow]File system is disabled.[/yellow]")
            return
        
        console.print(f"[dim]Searching for: {subarg}[/dim]")
        results = mech.search_files(subarg)
        
        if not results:
            console.print("[dim]No files found.[/dim]")
            return
        
        table = Table(title=f"Search Results for '{subarg}'", show_header=True)
        table.add_column("File", style="cyan")
        table.add_column("Size")
        table.add_column("Path")
        
        for f in results[:30]:
            table.add_row(f.name, f.size_human, str(f.path.parent))
        
        if len(results) > 30:
            console.print(f"[dim]... and {len(results) - 30} more results[/dim]")
        
        console.print(table)
    
    else:
        console.print(f"[red]Unknown files command: {subcmd}[/red]")
        console.print("Available: list, read, tree, search")


async def handle_web_command(args: str | None, mech: DAVM) -> None:
    """Handle web commands."""
    if not args:
        # Show web system status
        if not mech.web_enabled:
            console.print("[yellow]Web system is disabled.[/yellow]")
        else:
            console.print("[green]Web system is enabled.[/green]")
        return
    
    parts = args.split(maxsplit=1)
    subcmd = parts[0].lower()
    subarg = parts[1] if len(parts) > 1 else None
    
    if subcmd == "fetch":
        if not subarg:
            console.print("[red]Usage: /web fetch <url>[/red]")
            return
        
        if not mech.web_enabled:
            console.print("[yellow]Web system is disabled.[/yellow]")
            return
        
        console.print(f"[dim]Fetching: {subarg}[/dim]")
        result = await mech.fetch_url(subarg)
        
        if result.success and result.page:
            page = result.page
            console.print(f"\n[bold]{page.title or 'No title'}[/bold]")
            console.print(f"[dim]{page.url}[/dim]")
            console.print()
            
            # Show text content (truncated)
            text = page.text
            if len(text) > 1500:
                text = text[:1500] + "\n\n... [truncated]"
            console.print(Panel(text, border_style="green"))
        else:
            console.print(f"[red]Error: {result.message}[/red]")
    
    elif subcmd == "summary":
        if not subarg:
            console.print("[red]Usage: /web summary <url>[/red]")
            return
        
        if not mech.web_enabled:
            console.print("[yellow]Web system is disabled.[/yellow]")
            return
        
        console.print(f"[dim]Fetching summary: {subarg}[/dim]")
        summary = await mech.get_page_summary(subarg)
        
        if summary.get("success"):
            table = Table(title="Page Summary", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value")
            
            table.add_row("Title", summary.get("title", "N/A"))
            table.add_row("URL", summary.get("url", "N/A"))
            table.add_row("Domain", summary.get("domain", "N/A"))
            table.add_row("Description", summary.get("description", "N/A")[:100] + "..." if len(summary.get("description", "")) > 100 else summary.get("description", "N/A"))
            table.add_row("Text Length", f"{summary.get('text_length', 0):,} characters")
            table.add_row("Links", str(summary.get("link_count", 0)))
            table.add_row("Images", str(summary.get("image_count", 0)))
            
            console.print(table)
        else:
            console.print(f"[red]Error: {summary.get('error', 'Unknown error')}[/red]")
    
    elif subcmd == "links":
        if not subarg:
            console.print("[red]Usage: /web links <url>[/red]")
            return
        
        if not mech.web_enabled or not mech.web:
            console.print("[yellow]Web system is disabled.[/yellow]")
            return
        
        console.print(f"[dim]Fetching links from: {subarg}[/dim]")
        links = await mech.web.fetch_links(subarg)
        
        if not links:
            console.print("[dim]No links found.[/dim]")
            return
        
        table = Table(title=f"Links from {subarg}", show_header=True)
        table.add_column("Text", style="cyan", max_width=40)
        table.add_column("URL", max_width=60)
        
        for link in links[:30]:
            text = link.text[:40] if len(link.text) > 40 else link.text
            url = link.url[:60] if len(link.url) > 60 else link.url
            table.add_row(text, url)
        
        if len(links) > 30:
            console.print(f"[dim]... and {len(links) - 30} more links[/dim]")
        
        console.print(table)
    
    else:
        console.print(f"[red]Unknown web command: {subcmd}[/red]")
        console.print("Available: fetch, summary, links")


async def handle_command(command: str, mech: DAVM) -> bool:
    """
    Handle a CLI command.
    
    Returns True if should continue, False to exit.
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    
    if cmd in ("/exit", "/quit", "/q"):
        console.print("[yellow]Shutting down DAVM...[/yellow]")
        return False
    
    elif cmd == "/help":
        print_help()
    
    elif cmd == "/status":
        await print_status(mech)
    
    elif cmd == "/models":
        await print_models(mech, pilot=arg if arg in ("anthropic", "ollama") else None)
    
    elif cmd == "/switch":
        if not arg or arg not in ("anthropic", "ollama"):
            console.print("[red]Usage: /switch <anthropic|ollama>[/red]")
        else:
            try:
                console.print(f"[yellow]Switching to {arg} pilot...[/yellow]")
                await mech.switch_pilot(pilot=arg, preserve_history=True)
                console.print(f"[green]Switched to {arg} pilot![/green]")
                await print_status(mech)
            except Exception as e:
                console.print(f"[red]Failed to switch: {e}[/red]")
    
    elif cmd == "/model":
        if not arg:
            console.print("[red]Usage: /model <model-name>[/red]")
        else:
            try:
                pilot_type = mech.pilot_type
                console.print(f"[yellow]Switching to model {arg}...[/yellow]")
                await mech.activate(pilot=pilot_type, model=arg)
                console.print(f"[green]Now using model: {arg}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to switch model: {e}[/red]")
    
    elif cmd == "/autonomy":
        if not arg or arg not in ("ask", "semi", "full"):
            console.print("[red]Usage: /autonomy <ask|semi|full>[/red]")
            console.print("  ask  - Always ask permission")
            console.print("  semi - Ask for writes, read freely")
            console.print("  full - Fully autonomous")
        else:
            mech.set_autonomy_level(arg)
            console.print(f"[green]Autonomy level set to: {arg}[/green]")

    elif cmd == "/thinking":
        handle_thinking_command(arg)
    
    elif cmd == "/clear":
        mech.clear_history()
        console.print("[green]Conversation history cleared.[/green]")
        console.print("[dim](Long-term memories preserved. Use /memory clear to delete those.)[/dim]")
    
    elif cmd == "/history":
        print_history(mech)
    
    elif cmd == "/memory":
        await handle_memory_command(arg, mech)
    
    elif cmd == "/files":
        await handle_files_command(arg, mech)
    
    elif cmd == "/web":
        await handle_web_command(arg, mech)
    
    elif cmd == "/agent":
        return await handle_agent_command(arg, mech)
    
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print("Type [cyan]/help[/cyan] for available commands.")
    
    return True


# Global flag for agent mode
_agent_mode_enabled = False

# Global flag for thinking output
_show_thinking_enabled = False


async def handle_agent_command(args: str | None, mech: DAVM) -> bool:
    """Handle /agent command."""
    global _agent_mode_enabled
    
    if not args:
        # Show status
        status = "[green]ON[/green]" if _agent_mode_enabled else "[yellow]OFF[/yellow]"
        console.print(f"Agent mode: {status}")
        console.print("[dim]In agent mode, the LLM can use tools autonomously.[/dim]")
        console.print(f"Current autonomy level: [cyan]{mech.autonomy_level}[/cyan]")
        return True
    
    if args.lower() == "on":
        _agent_mode_enabled = True
        console.print("[green]Agent mode enabled.[/green]")
        console.print("The LLM will now use tools autonomously when helpful.")
        console.print(f"Autonomy level: [cyan]{mech.autonomy_level}[/cyan]")
        return True
    
    if args.lower() == "off":
        _agent_mode_enabled = False
        console.print("[yellow]Agent mode disabled.[/yellow]")
        console.print("Regular chat mode (streaming) restored.")
        return True
    
    # Otherwise, treat args as a one-off agent query
    console.print("[green bold]DAVM (Agent)>[/green bold]")
    await agentic_response(mech, args)
    return True


def is_agent_mode() -> bool:
    """Check if agent mode is enabled."""
    return _agent_mode_enabled


def handle_thinking_command(arg: str | None) -> None:
    """Handle /thinking command for showing/hiding thought output."""
    global _show_thinking_enabled

    if not arg:
        status = "ON" if _show_thinking_enabled else "OFF"
        console.print(f"Thinking output: [cyan]{status}[/cyan]")
        console.print("Use /thinking on or /thinking off")
        return

    if arg.lower() == "on":
        _show_thinking_enabled = True
        console.print("[green]Thinking output enabled.[/green]")
        return

    if arg.lower() == "off":
        _show_thinking_enabled = False
        console.print("[yellow]Thinking output disabled (spinner only).[/yellow]")
        return

    console.print("[red]Usage: /thinking <on|off>[/red]")


async def stream_response(mech: DAVM, message: str):
    """Stream a response from the mech with live display."""
    full_response = ""
    thinking_response = ""
    inside_thought = False
    pending_think_chunk = False
    
    try:
        def build_renderable() -> Group:
            parts = [Markdown(full_response or "")]
            if _show_thinking_enabled and thinking_response:
                parts.append(Text(f"Thinking:\n{thinking_response}", style="dim"))
            elif inside_thought:
                parts.append(Spinner("dots", text="Thinking..."))
            return Group(*parts)

        # Use live display for streaming
        with Live(console=console, refresh_per_second=10) as live:
            async for chunk in mech.chat_stream(message):
                if chunk == THINK_START_TOKEN:
                    inside_thought = True
                    pending_think_chunk = False
                    live.update(build_renderable())
                    continue

                if chunk == THINK_END_TOKEN:
                    inside_thought = False
                    pending_think_chunk = False
                    live.update(build_renderable())
                    continue

                if chunk == THINK_CHUNK_TOKEN:
                    pending_think_chunk = True
                    continue

                if pending_think_chunk:
                    if _show_thinking_enabled:
                        thinking_response += chunk
                    pending_think_chunk = False
                    live.update(build_renderable())
                    continue

                full_response += chunk
                live.update(build_renderable())
        
        # Final newline after streaming
        console.print()
        
        # Show memory indicator if memory was used
        if mech.memory_enabled:
            stats = await mech.get_memory_stats()
            count = stats.get("total_memories", 0)
            console.print(f"[dim](Memory: {count} memories stored)[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


async def agentic_response(mech: DAVM, message: str):
    """Get a response with tool use (agentic mode)."""
    try:
        console.print("[dim]Thinking with tools...[/dim]")
        
        response, tool_results = await mech.chat_with_tools(message)
        
        # Show tool usage
        if tool_results:
            console.print(f"[dim]Used {len(tool_results)} tool(s)[/dim]")
            for result in tool_results:
                status = "[green]OK[/green]" if result.success else "[red]FAIL[/red]"
                console.print(f"  {status} {result.tool_call_id}")
        
        # Show response
        console.print()
        console.print(Markdown(response))
        console.print()
        
        # Show memory indicator
        if mech.memory_enabled:
            stats = await mech.get_memory_stats()
            count = stats.get("total_memories", 0)
            console.print(f"[dim](Memory: {count} memories stored)[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


async def run_cli():
    """Run the main CLI loop."""
    print_banner()
    
    # Create the mech
    mech = DAVM()
    global _show_thinking_enabled
    _show_thinking_enabled = mech._settings.davm_show_thinking
    
    # Try to activate with default settings
    console.print("[yellow]Initializing DAVM...[/yellow]")
    
    try:
        await mech.activate()
        pilot_info = mech.get_pilot_info()
        apply_instruction_booklet(mech)
        console.print(
            f"[green]DAVM activated![/green] "
            f"Pilot: [cyan]{mech.pilot_type}[/cyan] | "
            f"Model: [cyan]{pilot_info.current_model if pilot_info else 'unknown'}[/cyan]"
        )
        
        # Show memory status
        if mech.memory_enabled:
            stats = await mech.get_memory_stats()
            console.print(
                f"[green]Memory enabled:[/green] {stats.get('total_memories', 0)} memories stored"
            )
        
    except Exception as e:
        console.print(f"[red]Failed to activate default pilot: {e}[/red]")
        console.print("[yellow]Trying Ollama as fallback...[/yellow]")
        try:
            await mech.activate(pilot="ollama")
            pilot_info = mech.get_pilot_info()
            apply_instruction_booklet(mech)
            console.print(
                f"[green]DAVM activated with Ollama![/green] "
                f"Model: [cyan]{pilot_info.current_model if pilot_info else 'unknown'}[/cyan]"
            )
            if mech.memory_enabled:
                stats = await mech.get_memory_stats()
                console.print(
                    f"[green]Memory enabled:[/green] {stats.get('total_memories', 0)} memories stored"
                )
        except Exception as e2:
            console.print(f"[red]Failed to activate Ollama: {e2}[/red]")
            console.print(
                "\n[yellow]Please ensure you have either:[/yellow]\n"
                "  1. Set ANTHROPIC_API_KEY in .env file\n"
                "  2. Ollama running with at least one model pulled\n"
            )
            return

    console.print("\nType [cyan]/help[/cyan] for commands, or just start chatting!")
    console.print("[dim]Your conversations are being remembered across sessions.[/dim]")
    console.print("─" * 60)
    
    # Set up prompt session with history
    history_file = mech._settings.davm_data_dir / ".cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        style=prompt_style,
    )
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = await session.prompt_async(
                [("class:prompt", "You> ")],
            )
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                should_continue = await handle_command(user_input, mech)
                if not should_continue:
                    break
                continue
            
            # Regular chat message
            if is_agent_mode():
                console.print("[green bold]DAVM (Agent)>[/green bold]")
                await agentic_response(mech, user_input)
            else:
                console.print("[green bold]DAVM>[/green bold]")
                await stream_response(mech, user_input)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit[/yellow]")
            continue
        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    # Cleanup
    await mech.deactivate()
    console.print("[cyan]DAVM shut down. Goodbye![/cyan]")


def main():
    """Entry point for the CLI."""
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        console.print("\n[cyan]DAVM shut down. Goodbye![/cyan]")
        sys.exit(0)


if __name__ == "__main__":
    main()
