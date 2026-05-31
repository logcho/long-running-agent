import os
import subprocess
import config

def get_safe_path(path):
    """
    Ensures that the path is absolute and within the workspace base directory.
    """
    abs_path = os.path.abspath(os.path.join(config.BASE_DIR, path))
    if not abs_path.startswith(config.BASE_DIR):
        # Fallback to config.BASE_DIR if trying to escape
        return os.path.join(config.BASE_DIR, os.path.basename(path))
    return abs_path

class ReadFileTool:
    name = "read_file"
    description = "Read the contents of a file in the workspace. path must be relative to workspace root."

    def __call__(self, path: str) -> str:
        safe_path = get_safe_path(path)
        if not os.path.exists(safe_path):
            return f"Error: File '{path}' does not exist."
        if os.path.isdir(safe_path):
            return f"Error: '{path}' is a directory. Use list_directory instead."
        try:
            with open(safe_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file '{path}': {e}"

class WriteFileTool:
    name = "write_file"
    description = "Write content to a file in the workspace. path must be relative to workspace root. Overwrites if file exists."

    def __call__(self, path: str, content: str) -> str:
        safe_path = get_safe_path(path)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        try:
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to file '{path}'"
        except Exception as e:
            return f"Error writing to file '{path}': {e}"

class ListDirectoryTool:
    name = "list_directory"
    description = "List files and folders in a workspace directory. path is relative to workspace root (use '.' for root)."

    def __call__(self, path: str = ".") -> str:
        safe_path = get_safe_path(path)
        if not os.path.exists(safe_path):
            return f"Error: Path '{path}' does not exist."
        if not os.path.isdir(safe_path):
            return f"Error: Path '{path}' is a file, not a directory."
        try:
            items = os.listdir(safe_path)
            # Filter out dotfiles and hidden folders (except config files if necessary)
            visible_items = [it for it in items if not it.startswith(".")]
            
            output = []
            for item in sorted(visible_items):
                item_path = os.path.join(safe_path, item)
                rel_path = os.path.relpath(item_path, config.BASE_DIR)
                if os.path.isdir(item_path):
                    output.append(f"[DIR]  {rel_path}/")
                else:
                    size = os.path.getsize(item_path)
                    output.append(f"[FILE] {rel_path} ({size} bytes)")
            
            if not output:
                return f"Directory '{path}' is empty."
            return "\n".join(output)
        except Exception as e:
            return f"Error listing directory '{path}': {e}"

class RunCommandTool:
    name = "run_command"
    description = "Run a terminal/bash command in the workspace. Commands are executed from the workspace root. Timeout is 30s."

    def __call__(self, command: str) -> str:
        # For security and stability, restrict path or environmental mutations
        # and enforce execution within config.BASE_DIR
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=config.BASE_DIR,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = []
            if result.returncode == 0:
                output.append(f"Command succeeded (exit code: 0).")
            else:
                output.append(f"Command failed (exit code: {result.returncode}).")
                
            if result.stdout:
                output.append(f"--- Standard Output ---\n{result.stdout}")
            if result.stderr:
                output.append(f"--- Standard Error ---\n{result.stderr}")
                
            if not result.stdout and not result.stderr:
                output.append("(No output produced)")
                
            return "\n".join(output)
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {e}"
