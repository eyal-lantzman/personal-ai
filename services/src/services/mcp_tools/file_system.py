from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class FileSystem:
    def __init__(self, root:str):
        if isinstance(root, Path):
            root = str(root)
        self.root = os.path.abspath(root) + os.sep
        self.mcp = FastMCP("File System for " + root)
        self.mcp.tool(self.list_dir)
        self.mcp.tool(self.read_text)
        self.mcp.tool(self.read_bytes)
        self.mcp.tool(self.write_to_file)
        self.mcp.tool(self.create_file)
        self.mcp.tool(self.create_folder)
        self.mcp.tool(self.is_folder_empty)
        self.mcp.tool(self.search_folder)
            
    def validate_relative_path(self, relative_path:str) -> Path:
        if relative_path == ".":
            relative_path = self.root
        elif relative_path.startswith("." + os.sep):
            relative_path = self.root + relative_path[1:]
        elif relative_path.startswith(".." + os.sep):
            relative_path = self.root + relative_path
        elif not os.path.isabs(relative_path):
            relative_path = self.root + relative_path

        other = os.path.abspath(relative_path)

        if not other.startswith(self.root) and other + os.sep != self.root:
            raise ValueError("Invalid relative path: %s", relative_path)
        
        return Path(other)

    def list_dir(self, 
                relative_path:Annotated[str, Field(description="Relative path from which to list the files and folder")]
                ) -> list[str]:
        """List directory files and directories for a 'path'.

        Args:
            relative_path (str): The relative path under which to list the files and folders.
            
        Returns:
            list[str]: List of files and folders under a 'path'.
        """
        absolute = self.validate_relative_path(relative_path)
        return os.listdir(str(absolute))

    def read_text(self,
                  file_name:Annotated[str, Field(description="File name to read the text from")],
                  relative_path:Annotated[str, Field(description="The relative path where the file is located")],
                  encoding:Annotated[str, Field(description="File encoding")] = None
                ) -> str:
        """Read content as string from 'file_name' under a 'relative_path' with specific 'encoding'.

        Args:
            file_name (str): The name of the file.
            relative_path (str): The relative path under which to create the folder
            encoding (str): Content encoding, optional.

        Returns:
            str: File text content.
        """
        
        absolute = self.validate_relative_path(relative_path)
        target = absolute / file_name
        return target.read_text(encoding=encoding)

    def read_bytes(self,
                   file_name:Annotated[str, Field(description="File name from which to read the bytes")],
                   relative_path:Annotated[str, Field(description="The relative path where the file is located")]
                ) -> bytes:
        """Read content as bytes from 'file_name' under a 'relative_path'.

        Args:
            file_name (str): The name of the file.
            relative_path (str): The relative path under which to create the folder

        Returns:
            bytes: File content as bytes.
        """
        
        absolute = self.validate_relative_path(relative_path)
        target = absolute / file_name
        return target.read_bytes()

    def write_to_file(self,
                      content:Annotated[str | bytes, Field(description="The content to write to the file")],
                      file_name:Annotated[str, Field(description="The file to which to write the content")],
                      relative_path:Annotated[str, Field(description="The relative path where the file is located")],
                      encoding:Annotated[str, Field(description="File encoding")] = None
                    ) -> int:
        """Writes content to 'file_name' under a 'relative_path'.
        Args:
            content (str | bytes): The content as string or bytes.
            file_name (str): The name of the file.
            relative_path (str): The relative path under which to create the folder
            encoding (str): Content encoding, optional.
        Returns:
            int: offset after write operation.
        """
        
        absolute = self.validate_relative_path(relative_path)
        target = absolute / file_name
        if isinstance(content, str):
            return target.write_text(content, encoding)
        else:
            return target.write_bytes(content)

    def create_file(self,
                    file_name:Annotated[str, Field(description="Name of the file to create")],
                    relative_path:Annotated[str, Field(description="The relative path where the file is located")],
                ) -> str:
        """Creates an empty file 'file_name' under a 'relative_path'.
        Args:
            file_name (str): The name of the file.
            relative_path (str): The relative path under which to create the folder.
        Returns:
            str: the full path for the file.
        """
        absolute = self.validate_relative_path(relative_path)
        if not absolute.exists() or not absolute.is_dir():
            raise ValueError("Unexpected relative path: %s", relative_path)
        target = absolute / file_name
        target.touch()
        return str(target)

    def create_folder(self, 
                      folder_name:Annotated[str, Field(description="Name of the folder to create")], 
                      relative_path:Annotated[str, Field(description="Relative path in which to create a folder")]
                    ) -> str:
        """Creates a folder 'folder_name' under a 'relative_path'.
        Args:
            folder_name (str): The name of the folder.
            relative_path (str): The relative path under which to create the folder.
        Returns:
            str: the full path for the created folder.
        """
        absolute = self.validate_relative_path(relative_path)
        if not absolute.exists() or not absolute.is_dir():
            raise ValueError("Unexpected relative path: %s", relative_path)
        target = absolute / folder_name
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def is_folder_empty(self,
                        relative_path:Annotated[str, Field(description="Relative to check if it's an empty folder")]
                    ) -> bool:
        """Checks if the folder 'folder_name' is empty.
        Args:
            relative_path (str): The name of the folder.
        Returns:
            bool: if folder is empty.
        """
        absolute = self.validate_relative_path(relative_path)
        if not absolute.exists() or not absolute.is_dir():
            raise ValueError("Unexpected path: %s", relative_path)
        return not any(absolute.glob("*"))

    def search_folder(self,
                      relative_path:Annotated[str, Field(description="Relative to check if it's an empty folder")],
                      pattern:Annotated[str, Field(description="Search pattern as a regular expression")]
                    ) -> list[str]:
        """Searches a pattern (file or folder) under 'folder_name'.
        Args:
            relative_path (str): The name of the folder to search from.
            pattern (str): The search pattern.
        Returns:
            bool: if folder is empty.
        """
        absolute = self.validate_relative_path(relative_path)
        if not absolute.exists() or not absolute.is_dir():
            raise ValueError("Unexpected path: %s", relative_path)
        return [str(p.relative_to(self.root)) for p in absolute.glob(pattern)]
