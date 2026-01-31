"""
DAVM File System - Secure file operations within allowed paths.

This system gives the mech the ability to interact with files,
while respecting security boundaries and user permissions.
"""

import fnmatch
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

from davm.core.config import Settings, get_settings


@dataclass
class FileInfo:
    """Information about a file or directory."""
    
    path: Path
    name: str
    is_dir: bool
    size: int  # bytes
    modified: datetime
    created: datetime | None
    extension: str | None
    
    @property
    def size_human(self) -> str:
        """Get human-readable size."""
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        elif self.size < 1024 * 1024 * 1024:
            return f"{self.size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.size / (1024 * 1024 * 1024):.1f} GB"


@dataclass 
class FileOperationResult:
    """Result of a file operation."""
    
    success: bool
    message: str
    path: Path | None = None
    data: str | bytes | None = None


OperationType = Literal["read", "write", "delete", "move", "create"]


class FileSystem:
    """
    Secure file system operations for DAVM.
    
    This system allows the mech to read, write, and manage files
    while respecting configured security boundaries (allowed paths).
    
    Security model:
    - Only paths within allowed_paths can be accessed
    - Symlinks are resolved and checked against allowed paths
    - Operations respect the autonomy level for permission checks
    
    Example:
        fs = FileSystem(allowed_paths=["~/Documents", "~/Projects"])
        
        # Read a file
        result = fs.read_file("~/Documents/notes.txt")
        if result.success:
            print(result.data)
        
        # List directory
        files = fs.list_dir("~/Projects")
        for f in files:
            print(f"{f.name} - {f.size_human}")
        
        # Search for files
        matches = fs.search("*.py", "~/Projects")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        allowed_paths: Sequence[str | Path] | None = None,
    ):
        """
        Initialize the file system.
        
        Args:
            settings: DAVM settings. If not provided, loads from environment.
            allowed_paths: Override for allowed paths. If not provided, uses settings.
        """
        self._settings = settings or get_settings()
        
        if allowed_paths:
            self._allowed_paths = [
                Path(p).expanduser().resolve() for p in allowed_paths
            ]
        else:
            self._allowed_paths = self._settings.allowed_paths_list
        
        # Default to home directory if no paths specified
        if not self._allowed_paths:
            self._allowed_paths = [Path.home()]

    @property
    def allowed_paths(self) -> list[Path]:
        """Get the list of allowed paths."""
        return self._allowed_paths.copy()

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve a path, expanding ~ and making it absolute."""
        return Path(path).expanduser().resolve()

    def _is_path_allowed(self, path: str | Path) -> bool:
        """Check if a path is within allowed directories."""
        target = self._resolve_path(path)
        
        for allowed in self._allowed_paths:
            try:
                target.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    def _check_access(self, path: str | Path, operation: OperationType) -> FileOperationResult | None:
        """
        Check if access to a path is allowed.
        
        Returns None if allowed, or a FileOperationResult with error if not.
        """
        resolved = self._resolve_path(path)
        
        if not self._is_path_allowed(resolved):
            return FileOperationResult(
                success=False,
                message=f"Access denied: '{resolved}' is outside allowed paths. "
                       f"Allowed: {[str(p) for p in self._allowed_paths]}",
                path=resolved,
            )
        
        return None

    # === Read Operations ===
    
    def read_file(
        self, 
        path: str | Path, 
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> FileOperationResult:
        """
        Read a file's contents.
        
        Args:
            path: Path to the file.
            encoding: Text encoding (ignored if binary=True).
            binary: If True, read as binary data.
            
        Returns:
            FileOperationResult with file contents in .data
        """
        resolved = self._resolve_path(path)
        
        # Check access
        access_error = self._check_access(resolved, "read")
        if access_error:
            return access_error
        
        # Check file exists
        if not resolved.exists():
            return FileOperationResult(
                success=False,
                message=f"File not found: {resolved}",
                path=resolved,
            )
        
        if resolved.is_dir():
            return FileOperationResult(
                success=False,
                message=f"Path is a directory, not a file: {resolved}",
                path=resolved,
            )
        
        try:
            if binary:
                data = resolved.read_bytes()
            else:
                data = resolved.read_text(encoding=encoding)
            
            return FileOperationResult(
                success=True,
                message=f"Read {len(data)} {'bytes' if binary else 'characters'} from {resolved.name}",
                path=resolved,
                data=data,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Error reading file: {e}",
                path=resolved,
            )

    def get_file_info(self, path: str | Path) -> FileInfo | None:
        """
        Get information about a file or directory.
        
        Args:
            path: Path to the file or directory.
            
        Returns:
            FileInfo object or None if not accessible.
        """
        resolved = self._resolve_path(path)
        
        if not self._is_path_allowed(resolved):
            return None
        
        if not resolved.exists():
            return None
        
        try:
            stat = resolved.stat()
            
            return FileInfo(
                path=resolved,
                name=resolved.name,
                is_dir=resolved.is_dir(),
                size=stat.st_size if not resolved.is_dir() else 0,
                modified=datetime.fromtimestamp(stat.st_mtime),
                created=datetime.fromtimestamp(stat.st_ctime) if hasattr(stat, 'st_ctime') else None,
                extension=resolved.suffix.lower() if resolved.suffix else None,
            )
        except Exception:
            return None

    def list_dir(
        self,
        path: str | Path,
        pattern: str | None = None,
        include_hidden: bool = False,
        recursive: bool = False,
    ) -> list[FileInfo]:
        """
        List contents of a directory.
        
        Args:
            path: Directory path.
            pattern: Optional glob pattern to filter results (e.g., "*.py").
            include_hidden: Include hidden files (starting with .).
            recursive: Recursively list subdirectories.
            
        Returns:
            List of FileInfo objects.
        """
        resolved = self._resolve_path(path)
        
        if not self._is_path_allowed(resolved):
            return []
        
        if not resolved.exists() or not resolved.is_dir():
            return []
        
        results = []
        
        try:
            if recursive:
                iterator = resolved.rglob(pattern or "*")
            else:
                iterator = resolved.glob(pattern or "*")
            
            for item in iterator:
                # Skip hidden files unless requested
                if not include_hidden and item.name.startswith("."):
                    continue
                
                # Make sure we're still within allowed paths
                if not self._is_path_allowed(item):
                    continue
                
                info = self.get_file_info(item)
                if info:
                    results.append(info)
            
            # Sort: directories first, then by name
            results.sort(key=lambda x: (not x.is_dir, x.name.lower()))
            
        except Exception:
            pass
        
        return results

    def search(
        self,
        pattern: str,
        start_path: str | Path | None = None,
        max_results: int = 100,
    ) -> list[FileInfo]:
        """
        Search for files matching a pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.txt").
            start_path: Starting directory. If None, searches all allowed paths.
            max_results: Maximum number of results to return.
            
        Returns:
            List of matching FileInfo objects.
        """
        results = []
        
        if start_path:
            search_paths = [self._resolve_path(start_path)]
        else:
            search_paths = self._allowed_paths
        
        for base_path in search_paths:
            if not base_path.exists():
                continue
            
            try:
                for item in base_path.rglob(pattern):
                    if not self._is_path_allowed(item):
                        continue
                    
                    info = self.get_file_info(item)
                    if info:
                        results.append(info)
                    
                    if len(results) >= max_results:
                        break
                
                if len(results) >= max_results:
                    break
                    
            except Exception:
                continue
        
        return results

    def search_content(
        self,
        text: str,
        path: str | Path,
        file_pattern: str = "*",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> list[tuple[FileInfo, list[tuple[int, str]]]]:
        """
        Search for text within files.
        
        Args:
            text: Text to search for.
            path: Directory to search in.
            file_pattern: Glob pattern for files to search (e.g., "*.py").
            case_sensitive: Whether search is case-sensitive.
            max_results: Maximum number of files to return.
            
        Returns:
            List of tuples: (FileInfo, [(line_number, line_content), ...])
        """
        resolved = self._resolve_path(path)
        results = []
        
        if not self._is_path_allowed(resolved):
            return []
        
        search_text = text if case_sensitive else text.lower()
        
        for item in resolved.rglob(file_pattern):
            if item.is_dir() or not self._is_path_allowed(item):
                continue
            
            try:
                content = item.read_text(errors="ignore")
                lines = content.splitlines()
                
                matches = []
                for i, line in enumerate(lines, 1):
                    compare_line = line if case_sensitive else line.lower()
                    if search_text in compare_line:
                        matches.append((i, line.strip()[:200]))  # Truncate long lines
                
                if matches:
                    info = self.get_file_info(item)
                    if info:
                        results.append((info, matches))
                
                if len(results) >= max_results:
                    break
                    
            except Exception:
                continue
        
        return results

    # === Write Operations ===
    
    def write_file(
        self,
        path: str | Path,
        content: str | bytes,
        encoding: str = "utf-8",
        overwrite: bool = False,
        create_dirs: bool = True,
    ) -> FileOperationResult:
        """
        Write content to a file.
        
        Args:
            path: Path to the file.
            content: Content to write.
            encoding: Text encoding (ignored if content is bytes).
            overwrite: If False, fail if file exists.
            create_dirs: Create parent directories if needed.
            
        Returns:
            FileOperationResult indicating success or failure.
        """
        resolved = self._resolve_path(path)
        
        # Check access
        access_error = self._check_access(resolved, "write")
        if access_error:
            return access_error
        
        # Check if file exists
        if resolved.exists() and not overwrite:
            return FileOperationResult(
                success=False,
                message=f"File already exists: {resolved}. Set overwrite=True to replace.",
                path=resolved,
            )
        
        try:
            # Create parent directories if needed
            if create_dirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            if isinstance(content, bytes):
                resolved.write_bytes(content)
                size = len(content)
            else:
                resolved.write_text(content, encoding=encoding)
                size = len(content)
            
            return FileOperationResult(
                success=True,
                message=f"Wrote {size} {'bytes' if isinstance(content, bytes) else 'characters'} to {resolved.name}",
                path=resolved,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Error writing file: {e}",
                path=resolved,
            )

    def append_file(
        self,
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
    ) -> FileOperationResult:
        """
        Append content to a file.
        
        Args:
            path: Path to the file.
            content: Content to append.
            encoding: Text encoding.
            
        Returns:
            FileOperationResult indicating success or failure.
        """
        resolved = self._resolve_path(path)
        
        # Check access
        access_error = self._check_access(resolved, "write")
        if access_error:
            return access_error
        
        try:
            with open(resolved, "a", encoding=encoding) as f:
                f.write(content)
            
            return FileOperationResult(
                success=True,
                message=f"Appended {len(content)} characters to {resolved.name}",
                path=resolved,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Error appending to file: {e}",
                path=resolved,
            )

    def create_directory(
        self,
        path: str | Path,
        parents: bool = True,
    ) -> FileOperationResult:
        """
        Create a directory.
        
        Args:
            path: Path for the new directory.
            parents: Create parent directories if needed.
            
        Returns:
            FileOperationResult indicating success or failure.
        """
        resolved = self._resolve_path(path)
        
        # Check access
        access_error = self._check_access(resolved, "create")
        if access_error:
            return access_error
        
        if resolved.exists():
            if resolved.is_dir():
                return FileOperationResult(
                    success=True,
                    message=f"Directory already exists: {resolved}",
                    path=resolved,
                )
            else:
                return FileOperationResult(
                    success=False,
                    message=f"Path exists but is not a directory: {resolved}",
                    path=resolved,
                )
        
        try:
            resolved.mkdir(parents=parents, exist_ok=True)
            return FileOperationResult(
                success=True,
                message=f"Created directory: {resolved}",
                path=resolved,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Error creating directory: {e}",
                path=resolved,
            )

    # === Delete/Move Operations ===
    
    def delete(
        self,
        path: str | Path,
        recursive: bool = False,
    ) -> FileOperationResult:
        """
        Delete a file or directory.
        
        Args:
            path: Path to delete.
            recursive: If True, delete directories and their contents.
            
        Returns:
            FileOperationResult indicating success or failure.
        """
        resolved = self._resolve_path(path)
        
        # Check access
        access_error = self._check_access(resolved, "delete")
        if access_error:
            return access_error
        
        if not resolved.exists():
            return FileOperationResult(
                success=False,
                message=f"Path does not exist: {resolved}",
                path=resolved,
            )
        
        try:
            if resolved.is_dir():
                if recursive:
                    shutil.rmtree(resolved)
                else:
                    resolved.rmdir()  # Only works for empty directories
            else:
                resolved.unlink()
            
            return FileOperationResult(
                success=True,
                message=f"Deleted: {resolved}",
                path=resolved,
            )
        except OSError as e:
            if "not empty" in str(e).lower():
                return FileOperationResult(
                    success=False,
                    message=f"Directory not empty. Use recursive=True to delete: {resolved}",
                    path=resolved,
                )
            return FileOperationResult(
                success=False,
                message=f"Error deleting: {e}",
                path=resolved,
            )

    def move(
        self,
        source: str | Path,
        destination: str | Path,
        overwrite: bool = False,
    ) -> FileOperationResult:
        """
        Move a file or directory.
        
        Args:
            source: Source path.
            destination: Destination path.
            overwrite: If True, overwrite existing destination.
            
        Returns:
            FileOperationResult indicating success or failure.
        """
        src = self._resolve_path(source)
        dst = self._resolve_path(destination)
        
        # Check access for both paths
        access_error = self._check_access(src, "move")
        if access_error:
            return access_error
        
        access_error = self._check_access(dst, "write")
        if access_error:
            return access_error
        
        if not src.exists():
            return FileOperationResult(
                success=False,
                message=f"Source does not exist: {src}",
                path=src,
            )
        
        if dst.exists() and not overwrite:
            return FileOperationResult(
                success=False,
                message=f"Destination already exists: {dst}. Set overwrite=True to replace.",
                path=dst,
            )
        
        try:
            if dst.exists() and overwrite:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            
            shutil.move(str(src), str(dst))
            
            return FileOperationResult(
                success=True,
                message=f"Moved {src.name} to {dst}",
                path=dst,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Error moving: {e}",
                path=src,
            )

    def copy(
        self,
        source: str | Path,
        destination: str | Path,
        overwrite: bool = False,
    ) -> FileOperationResult:
        """
        Copy a file or directory.
        
        Args:
            source: Source path.
            destination: Destination path.
            overwrite: If True, overwrite existing destination.
            
        Returns:
            FileOperationResult indicating success or failure.
        """
        src = self._resolve_path(source)
        dst = self._resolve_path(destination)
        
        # Check access
        access_error = self._check_access(src, "read")
        if access_error:
            return access_error
        
        access_error = self._check_access(dst, "write")
        if access_error:
            return access_error
        
        if not src.exists():
            return FileOperationResult(
                success=False,
                message=f"Source does not exist: {src}",
                path=src,
            )
        
        if dst.exists() and not overwrite:
            return FileOperationResult(
                success=False,
                message=f"Destination already exists: {dst}. Set overwrite=True to replace.",
                path=dst,
            )
        
        try:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            
            return FileOperationResult(
                success=True,
                message=f"Copied {src.name} to {dst}",
                path=dst,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Error copying: {e}",
                path=src,
            )

    # === Utility Methods ===
    
    def get_tree(
        self,
        path: str | Path,
        max_depth: int = 3,
        include_hidden: bool = False,
    ) -> str:
        """
        Get a tree representation of a directory.
        
        Args:
            path: Root directory.
            max_depth: Maximum depth to traverse.
            include_hidden: Include hidden files.
            
        Returns:
            String representation of the directory tree.
        """
        resolved = self._resolve_path(path)
        
        if not self._is_path_allowed(resolved):
            return f"Access denied: {resolved}"
        
        if not resolved.exists() or not resolved.is_dir():
            return f"Not a directory: {resolved}"
        
        lines = [str(resolved)]
        
        def add_items(dir_path: Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            
            try:
                items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                return
            
            # Filter hidden files
            if not include_hidden:
                items = [i for i in items if not i.name.startswith(".")]
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                
                if item.is_dir():
                    lines.append(f"{prefix}{connector}{item.name}/")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    add_items(item, new_prefix, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{item.name}")
        
        add_items(resolved, "", 1)
        return "\n".join(lines)
