import json
from langgraph.graph import StateGraph, END

# Import configuration
import config
from src.state import AgentState
from src.llm_client import get_llm_client

# Import tools
from src.tools.rag_tool import RAGSearchTool, RAGAddDocumentTool
from src.tools.memory_tool import RetrieveMemoryTool, StoreMemoryTool, StoreLessonTool, memory_manager
from src.tools.workspace_tools import ReadFileTool, WriteFileTool, ListDirectoryTool, RunCommandTool

# Define FinishTaskTool inline for compilation
class FinishTaskTool:
    name = "finish_task"
    description = "Signal that the task is fully completed. summary is a detailed summary of what was accomplished."
    def __call__(self, summary: str) -> str:
        return f"Task marked as completed. Summary: {summary}"

def agent_node(state: AgentState):
    """
    Agent Node: Calls the local model with the current context, 
    persistent memories, and plans to decide the next action.
    """
    iterations = state.get("iterations", 0) + 1
    print(f"\n================ Agent Loop Iteration {iterations} ================")
    
    # Reload memory to get the latest state
    memory_manager.load_memory()
    lessons = memory_manager.memory.get("lessons_learned", [])
    facts = memory_manager.memory.get("facts", {})
    
    # Combine state-level lessons and memory-level lessons
    all_lessons = list(set(state.get("lessons_learned", []) + lessons))
    
    # Construct System Instructions
    system_prompt = f"""You are a long-running agent designed to persistently work until you complete the user's task.
Your task is: "{state['task']}"

--- CONTEXT & SUMMARY OF PRIOR WORK ---
{state['summary'] or "No prior work has been summarized yet. You are starting fresh."}

--- PERSISTENT MEMORIES & LESSONS LEARNED ---
Instructions from past experience (follow these carefully to avoid repeating mistakes):
{chr(10).join("- " + l for l in all_lessons) if all_lessons else "None yet."}

Stored Facts:
{json.dumps(facts, indent=2)}

--- WORKING MEMORY (SHORT TERM) ---
{json.dumps(state['working_memory'], indent=2)}

--- CURRENT PLAN & CHECKS ---
Current Plan:
{state['current_plan'] or "No plan established yet. Your first action should be to inspect the workspace, list the directory, and formulate a step-by-step plan using the store_memory tool or writing it in your reasoning."}

--- AVAILABLE TOOLS ---
You can call the following tools:
1. `rag_search(query: str, top_k: int)`: Search local RAG documents.
2. `rag_add_document(filename: str, content: str)`: Add a text/markdown file to the RAG knowledge base.
3. `retrieve_memory(query: str)`: Retrieve stored memories.
4. `store_memory(key: str, value: str)`: Store a key-value fact in memory.
5. `store_lesson(lesson: str)`: Store a self-improvement lesson learned.
6. `read_file(path: str)`: Read file content in the workspace (path is relative to workspace root).
7. `write_file(path: str, content: str)`: Write file content in the workspace.
8. `list_directory(path: str)`: List files in a workspace directory.
9. `run_command(command: str)`: Run shell commands in the workspace root.
10. `finish_task(summary: str)`: Call this tool when you have fully completed the task.

--- RESPONSE FORMAT ---
You must output a reasoning thought process first, followed by a tool call.
If you want to call a tool, you can do so natively. If your environment does not support native tool calling, output a markdown JSON block:
```json
{{
  "tool": "tool_name",
  "arguments": {{
    "arg1": "val1"
  }}
}}
```
Only call ONE tool per turn. If you want to finish the task, you MUST call the `finish_task` tool.
"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    # We define tool metadata for the LLM
    tools_list = [
        RAGSearchTool(),
        RAGAddDocumentTool(),
        RetrieveMemoryTool(),
        StoreMemoryTool(),
        StoreLessonTool(),
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        RunCommandTool(),
        FinishTaskTool()
    ]
    
    client = get_llm_client()
    content, tool_calls = client.chat(messages, tools=tools_list)
    
    print(f"\n[Agent Thoughts]:\n{content}")
    if tool_calls:
        print(f"[Agent Decided Tool Calls]: {tool_calls}")
    else:
        print("[Agent Decided Tool Calls]: None")

    # Add the assistant response to the message history
    assistant_msg = {
        "role": "assistant",
        "content": content
    }
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
        
    new_messages = state["messages"] + [assistant_msg]
    
    # Try to parse if agent wrote a new plan in its thoughts, or if it uses store_memory
    # For robust plan tracking, if the agent updates "plan" key in store_memory, we'll sync it
    # We can inspect tool calls to update current_plan and working_memory before execution
    current_plan = state.get("current_plan", "")
    working_memory = state.get("working_memory", {}).copy()
    
    for tc in tool_calls:
        func_info = tc.get("function", {})
        name = func_info.get("name")
        args = func_info.get("arguments", {})
        if name == "store_memory":
            k = args.get("key")
            v = args.get("value")
            if k == "plan" or k == "current_plan":
                current_plan = v
            else:
                working_memory[k] = v
        elif name == "store_lesson":
            lesson = args.get("lesson")
            if lesson and lesson not in all_lessons:
                all_lessons.append(lesson)

    return {
        "messages": new_messages,
        "iterations": iterations,
        "current_plan": current_plan,
        "working_memory": working_memory,
        "lessons_learned": all_lessons
    }

def action_node(state: AgentState):
    """
    Action Node: Executes the tool calls emitted by the agent node.
    """
    messages = state["messages"]
    last_msg = messages[-1] if messages else {}
    tool_calls = last_msg.get("tool_calls", [])
    
    new_messages = []
    steps_taken = state.get("steps_taken", [])[:]
    finished = state.get("finished", False)
    
    for tc in tool_calls:
        func_info = tc.get("function", {})
        name = func_info.get("name")
        args = func_info.get("arguments", {})
        call_id = tc.get("id") or f"call_{name}_{state['iterations']}"
        print(f"\n[Tool Action]: Executing {name}...")
        
        result = ""
        try:
            if name == "rag_search":
                result = RAGSearchTool()(**args)
            elif name == "rag_add_document":
                result = RAGAddDocumentTool()(**args)
            elif name == "retrieve_memory":
                result = RetrieveMemoryTool()(**args)
            elif name == "store_memory":
                result = StoreMemoryTool()(**args)
            elif name == "store_lesson":
                result = StoreLessonTool()(**args)
            elif name == "read_file":
                result = ReadFileTool()(**args)
            elif name == "write_file":
                result = WriteFileTool()(**args)
            elif name == "list_directory":
                result = ListDirectoryTool()(**args)
            elif name == "run_command":
                result = RunCommandTool()(**args)
            elif name == "finish_task":
                result = FinishTaskTool()(**args)
                finished = True
            else:
                result = f"Error: Tool '{name}' is not recognized."
        except Exception as e:
            result = f"Error executing tool '{name}': {e}"
            
        print(f"[Tool Result]: (truncated to 150 chars)\n{str(result)[:150]}...")
        
        new_messages.append({
            "role": "tool",
            "name": name,
            "tool_call_id": call_id,
            "content": str(result)
        })
        steps_taken.append(f"Iteration {state['iterations']}: Ran {name} -> {str(result)[:100]}...")
        
    return {
        "messages": state["messages"] + new_messages,
        "steps_taken": steps_taken,
        "finished": finished
    }

def summarize_node(state: AgentState):
    """
    Summarize Node: Triggers when messages context is full.
    Compresses conversational history into state['summary'] and clears state['messages'].
    """
    print("\n--- [Context Compression] Active context window limit reached! Summarizing history... ---")
    messages = state["messages"]
    
    summary_prompt = f"""You are a helper summarizing the work of a long-running agent working on: "{state['task']}"
Prior summary of earlier work: {state['summary'] or 'None'}

Please read the chat history below and write a detailed, consolidated summary. Include:
1. What has been accomplished so far.
2. What files have been modified or created.
3. What facts, paths, or settings have been discovered.
4. What the current status is and what needs to be done next.

Chat History:
{json.dumps(messages, indent=2)}

Write your summary now:"""

    client = get_llm_client()
    summary_content, _ = client.chat(messages=[{"role": "user", "content": summary_prompt}])
    
    print(f"\n[Summarization Result]:\n{summary_content}\n")
    
    # We prune active messages to prevent context overflow.
    # Keep the last 2 messages for immediate operational continuity (e.g., tool result).
    retained_messages = messages[-2:] if len(messages) >= 2 else messages
    
    # Add a system injection message to let the agent know what happened
    info_msg = {
        "role": "system",
        "content": f"[System Note: Context has been compressed to prevent token limit overflow. Summary of previous operations: {summary_content}]"
    }
    
    return {
        "summary": summary_content,
        "messages": [info_msg] + retained_messages
    }

def improve_node(state: AgentState):
    """
    Improve Node: Performs final reflection on the agent's actions, 
    distills lessons learned, and updates persistent memory.
    """
    print("\n--- [Self-Improvement] Running performance evaluation & self-reflection... ---")
    
    steps = state.get("steps_taken", [])
    reflection_prompt = f"""You have completed (or are stopping) the task: "{state['task']}"

Summary of prior work: {state['summary'] or 'None'}
Steps taken in this run:
{json.dumps(steps, indent=2)}

Please reflect on this run:
1. What worked well and was efficient?
2. What mistakes did you make, or what took too many steps?
3. Formulate 1-2 general principles/lessons learned that you should follow in the future to complete similar tasks more quickly or avoid errors.

Format your lessons learned as clean, actionable instructions (e.g. "When compiling a python file, make sure to execute with the appropriate virtualenv interpreter").

Lessons Learned:"""

    client = get_llm_client()
    reflection_content, _ = client.chat(messages=[{"role": "user", "content": reflection_prompt}])
    
    print(f"\n[Self-Reflection Lessons Learned]:\n{reflection_content}\n")
    
    # Store the reflection in memory
    memory_manager.append_lesson(reflection_content)
    
    # Update facts to memory
    for k, v in state.get("working_memory", {}).items():
        memory_manager.store_fact(k, v)
        
    # Log completed task
    memory_manager.log_completed_task(state["task"], state.get("summary", "") + "\n" + reflection_content)
    
    return {
        "lessons_learned": memory_manager.memory.get("lessons_learned", []),
        "finished": True
    }

# Build LangGraph Workflow Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("action", action_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("improve", improve_node)

# Set Entry Point
workflow.set_entry_point("agent")

# Define Routing Edges
def route_after_agent(state: AgentState):
    if state.get("finished"):
        return "improve"
    
    messages = state["messages"]
    last_msg = messages[-1] if messages else {}
    tool_calls = last_msg.get("tool_calls", [])
    
    if tool_calls:
        return "action"
    
    if state.get("iterations", 0) >= config.MAX_ITERATIONS:
        print("\n[System Warning]: Max iterations reached! Going to self-reflection.")
        return "improve"
        
    # If no tool calls, and not finished, go to improve
    return "improve"

def route_after_action(state: AgentState):
    # Check if active messages list is too crowded
    if len(state["messages"]) >= config.CONTEXT_WINDOW_LIMIT:
        return "summarize"
    return "agent"

# Set Conditional Edges
workflow.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "action": "action",
        "improve": "improve"
    }
)

workflow.add_conditional_edges(
    "action",
    route_after_action,
    {
        "summarize": "summarize",
        "agent": "agent"
    }
)

workflow.add_edge("summarize", "agent")
workflow.add_edge("improve", END)

# Compile the graph
agent_app = workflow.compile()
