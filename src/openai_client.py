import os
import json
import re
import uuid
from openai import OpenAI
import config

class OpenAIClient:
    def __init__(self):
        api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or config value.")
        self.client = OpenAI(api_key=api_key)
        self.model = config.OPENAI_MODEL

    def chat(self, messages, tools=None):
        """
        Sends a message history to OpenAI.
        """
        formatted_tools = []
        if tools:
            for t in tools:
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

        # Sanitize/adapt messages for OpenAI
        # 1. OpenAI requires tool messages to have tool_call_id.
        # 2. Stringify arguments for OpenAI native assistant tool_calls.
        adapted_messages = []
        for m in messages:
            msg_copy = m.copy()
            
            # Map role
            role = msg_copy.get("role")
            
            # OpenAI requires arguments in assistant tool calls to be JSON strings
            if role == "assistant" and "tool_calls" in msg_copy:
                adapted_tc = []
                for tc in msg_copy["tool_calls"]:
                    tc_copy = tc.copy()
                    func_copy = tc_copy["function"].copy()
                    if isinstance(func_copy.get("arguments"), dict):
                        func_copy["arguments"] = json.dumps(func_copy["arguments"])
                    tc_copy["function"] = func_copy
                    adapted_tc.append(tc_copy)
                msg_copy["tool_calls"] = adapted_tc
                
            # OpenAI requires tool message to contain tool_call_id
            if role == "tool":
                # Ensure tool_call_id is set
                if "tool_call_id" not in msg_copy:
                    # Fallback ID if missing
                    msg_copy["tool_call_id"] = f"call_{msg_copy.get('name', 'tool')}"
                # Remove extra keys that OpenAI doesn't allow in tool messages
                msg_copy = {
                    "role": "tool",
                    "tool_call_id": msg_copy["tool_call_id"],
                    "content": msg_copy["content"]
                }
                
            adapted_messages.append(msg_copy)

        try:
            # Prepare arguments
            kwargs = {
                "model": self.model,
                "messages": adapted_messages,
            }
            if formatted_tools:
                kwargs["tools"] = formatted_tools

            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            tool_calls = message.tool_calls

            # Track tokens
            usage = getattr(response, "usage", None)
            if usage:
                from src.token_tracker import TokenTracker
                TokenTracker.add(usage.prompt_tokens, usage.completion_tokens, provider="openai")

            parsed_calls = []
            
            # 1. Parse native tool calls
            if tool_calls:
                for tc in tool_calls:
                    func_info = tc.function
                    try:
                        args_dict = json.loads(func_info.arguments) if func_info.arguments else {}
                    except json.JSONDecodeError:
                        args_dict = {"raw_arguments": func_info.arguments}
                        
                    parsed_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": func_info.name,
                            "arguments": args_dict
                        }
                    })
            
            # 2. Fallback: Parse markdown JSON blocks if the model tried to call tools via text
            if not parsed_calls and content:
                fallback_calls = self._parse_json_fallback(content)
                for fc in fallback_calls:
                    parsed_calls.append({
                        "id": f"call_{str(uuid.uuid4())[:8]}",
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": fc["args"]
                        }
                    })

            return content, parsed_calls

        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise e

    def get_embedding(self, text):
        """
        Generates text embedding using OpenAI's embedding API.
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            return []

    def _parse_json_fallback(self, content):
        tool_calls = []
        json_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        matches = json_pattern.findall(content)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, dict):
                    if "tool" in data:
                        tool_calls.append({
                            "name": data["tool"],
                            "args": data.get("arguments", {})
                        })
                    elif "tool_calls" in data and isinstance(data["tool_calls"], list):
                        for tc in data["tool_calls"]:
                            if "tool" in tc:
                                tool_calls.append({
                                    "name": tc["tool"],
                                    "args": tc.get("arguments", {})
                                })
            except json.JSONDecodeError:
                continue

        # Look for custom XML/bracket tags: <tool_call name="tool_name">{"arg1": "val1"}</tool_call>
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
