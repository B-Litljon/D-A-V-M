"""Basic tests for DAVM core functionality."""

import pytest
import pytest_asyncio
from pathlib import Path

from davm import DAVM
from davm.core.config import Settings
from davm.systems.filesystem import FileSystem
from davm.systems.tools import Tool, ToolCategory, ToolParameter, get_tool_by_name


class TestDAVMBasics:
    """Test basic DAVM initialization and properties."""

    def test_davm_init(self):
        """Test DAVM can be initialized."""
        mech = DAVM()
        assert mech is not None
        assert not mech.active
        assert mech.pilot is None
        assert mech.autonomy_level in ("ask", "semi", "full")  # Valid autonomy level

    def test_davm_autonomy_levels(self):
        """Test setting autonomy levels."""
        mech = DAVM()
        
        mech.set_autonomy_level("ask")
        assert mech.autonomy_level == "ask"
        
        mech.set_autonomy_level("semi")
        assert mech.autonomy_level == "semi"
        
        mech.set_autonomy_level("full")
        assert mech.autonomy_level == "full"


class TestFileSystem:
    """Test file system operations."""

    def test_filesystem_init(self):
        """Test FileSystem can be initialized."""
        fs = FileSystem()
        assert fs is not None

    def test_filesystem_path_validation(self, tmp_path: Path):
        """Test that FileSystem respects allowed paths."""
        # Create a filesystem limited to tmp_path
        fs = FileSystem(allowed_paths=[tmp_path])
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        
        # Should be able to read from allowed path
        result = fs.read_file(str(test_file))
        assert result.success
        assert result.data == "hello world"

    def test_filesystem_list_dir(self, tmp_path: Path):
        """Test directory listing."""
        fs = FileSystem(allowed_paths=[tmp_path])
        
        # Create some test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")
        (tmp_path / "subdir").mkdir()
        
        files = fs.list_dir(str(tmp_path))
        names = [f.name for f in files]
        
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "subdir" in names


class TestTools:
    """Test tool system."""

    def test_tool_definition(self):
        """Test tool definition structure."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.MEMORY,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
            ],
        )
        
        assert tool.name == "test_tool"
        assert tool.category == ToolCategory.MEMORY
        assert len(tool.parameters) == 1

    def test_tool_to_anthropic_schema(self):
        """Test converting tool to Anthropic format."""
        tool = Tool(
            name="search",
            description="Search for something",
            category=ToolCategory.MEMORY,
            parameters=[
                ToolParameter(name="q", type="string", description="Query"),
            ],
        )
        
        schema = tool.to_anthropic_schema()
        
        assert schema["name"] == "search"
        assert schema["description"] == "Search for something"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "q" in schema["input_schema"]["properties"]

    def test_tool_to_ollama_schema(self):
        """Test converting tool to Ollama (OpenAI) format."""
        tool = Tool(
            name="search",
            description="Search for something",
            category=ToolCategory.MEMORY,
            parameters=[
                ToolParameter(name="q", type="string", description="Query"),
            ],
        )
        
        schema = tool.to_ollama_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "parameters" in schema["function"]

    def test_get_tool_by_name(self):
        """Test looking up tools by name."""
        # These tools should exist in the default set
        memory_tool = get_tool_by_name("search_memory")
        assert memory_tool is not None
        assert memory_tool.category == ToolCategory.MEMORY
        
        file_tool = get_tool_by_name("read_file")
        assert file_tool is not None
        assert file_tool.category == ToolCategory.FILES_READ
        
        # Non-existent tool
        assert get_tool_by_name("nonexistent_tool") is None


@pytest.mark.asyncio
class TestDAVMAsync:
    """Test async DAVM operations (requires Ollama running)."""

    async def test_activate_ollama(self):
        """Test activating with Ollama (skip if Ollama not available)."""
        mech = DAVM()
        
        try:
            await mech.activate(
                pilot="ollama",
                model="llama3.2",
                enable_memory=False,
                enable_files=False,
                enable_web=False,
            )
            
            assert mech.active
            assert mech.pilot_type == "ollama"
            
            await mech.deactivate()
            assert not mech.active
            
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    async def test_basic_chat(self):
        """Test basic chat functionality."""
        mech = DAVM()
        
        try:
            await mech.activate(
                pilot="ollama",
                model="llama3.2",
                enable_memory=False,
            )
            
            result = await mech.chat("Say 'test' and nothing else.")
            assert result is not None
            assert len(result.content) > 0
            
            await mech.deactivate()
            
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
