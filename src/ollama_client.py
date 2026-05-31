import json
import re
from ollama import Client
import config

class OllamaClient:
    def __init__(self):
        self.client = Client(host=config.OLLAMA_BASE_URL)
        self.model = config.OLLAMA_MODEL

    def chat(self, messages, tools=None):
        """
        Sends a message history to Ollama.
        If tools are provided, it attempts to pass them to Ollama.
        It returns (response_content, parsed_tool_calls).
        """
        formatted_tools = []
        if tools:
            for t in tools:
                # Format to Ollama's expected schema
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.args_schema.schema() if hasattr(t, "args_schema") and t.args_schema else {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                })

        try:
            # Prepare request arguments
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if formatted_tools:
                kwargs["tools"] = formatted_tools

            response = self.client.chat(**kwargs)
            message = response.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            parsed_calls = []
            
            # 1. Parse native tool calls if present
            if tool_calls:
                for tc in tool_calls:
                    tc_dict = dict(tc) if not isinstance(tc, dict) else tc
                    func_dict = tc_dict.get("function", {})
                    func_dict = dict(func_dict) if not isinstance(func_dict, dict) else func_dict
                    
                    parsed_calls.append({
                        "type": "function",
                        "function": {
                            "name": func_dict.get("name"),
                            "arguments": func_dict.get("arguments", {})
                        }
                    })
            
            # 2. Fallback: Parse markdown JSON blocks if the model tried to call tools via text
            if not parsed_calls and content:
                fallback_calls = self._parse_json_fallback(content)
                for fc in fallback_calls:
                    parsed_calls.append({
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": fc["args"]
                        }
                    })

            return content, parsed_calls

        except Exception as e:
            print(f"Ollama API error: {e}")
            raise e

    def get_embedding(self, text):
        """
        Generates text embedding vector using the Ollama model.
        """
        try:
            response = self.client.embeddings(model=self.model, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            print(f"Ollama embedding error: {e}")
            # Return dummy embeddings in case the model does not support it
            # (or we fallback to basic TF-IDF in RAG implementation)
            return []

    def _parse_json_fallback(self, content):
        """
        Helper to parse JSON tool calls when local models fail to use native tool calling.
        Expects:
        ```json
        {
          "tool": "tool_name",
          "arguments": { ... }
        }
        ```
        or similar structures.
        """
        tool_calls = []
        # Find JSON blocks
        json_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        matches = json_pattern.findall(content)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, dict):
                    # Check if single tool structure
                    if "tool" in data:
                        tool_calls.append({
                            "name": data["tool"],
                            "args": data.get("arguments", {})
                        })
                    # Check if list of tool structures
                    elif "tool_calls" in data and isinstance(data["tool_calls"], list):
                        for tc in data["tool_calls"]:
                            if "tool" in tc:
                                tool_calls.append({
                                    "name": tc["tool"],
                                    "args": tc.get("arguments", {})
                                })
            except json.JSONDecodeError:
                continue

        # Look for custom XML/bracket tags as second fallback: <tool_call name="tool_name">{"arg1": "val1"}</tool_call>
        xml_pattern = re.compile(r'<tool_call\s+name="([^"]+)">\s*(.*?)\s*</tool_call>', re.DOTALL)
        xml_matches = xml_pattern.findall(content)
        for name, args_str in xml_matches:
            try:
                args = json.loads(args_str.strip())
                tool_calls.append({
                    "name": name,
                    "args": args
                })
            except json.JSONDecodeError:
                tool_calls.append({
                    "name": name,
                    "args": {"raw_input": args_str.strip()}
                })

        return tool_calls
