import sys
import argparse
import json
import config
from src.agent import agent_app
from src.tools.memory_tool import memory_manager
from src.tools.rag_tool import rag_system

def run_agent(task_text, context_limit=None, max_iter=None):
    # Override configuration if provided via CLI
    if context_limit is not None:
        config.CONTEXT_WINDOW_LIMIT = context_limit
        print(f"[Config] Override: CONTEXT_WINDOW_LIMIT = {context_limit}")
    if max_iter is not None:
        config.MAX_ITERATIONS = max_iter
        print(f"[Config] Override: MAX_ITERATIONS = {max_iter}")

    # Reindex RAG just in case new documents were added before start
    rag_system.reindex()

    # Load lessons and facts from memory manager
    memory_manager.load_memory()
    lessons = memory_manager.memory.get("lessons_learned", [])
    
    print("\n" + "="*50)
    print("      STARTING LONG-RUNNING LANGGRAPH AGENT      ")
    print("="*50)
    print(f"Goal: {task_text}")
    print(f"Loaded {len(lessons)} past lessons learned from memory.")
    print("="*50 + "\n")

    # Construct initial state
    initial_state = {
        "messages": [],
        "task": task_text,
        "summary": "",
        "lessons_learned": lessons,
        "working_memory": {},
        "current_plan": "",
        "steps_taken": [],
        "iterations": 0,
        "finished": False
    }

    # Run the graph
    # We can stream the steps of the graph to display what node is running
    try:
        final_state = None
        for step in agent_app.stream(initial_state, {"recursion_limit": config.MAX_ITERATIONS * 3}):
            for node_name, state_update in step.items():
                print(f"\n>>> Node Finished: {node_name}")
                if "summary" in state_update and state_update["summary"]:
                    print(f"[Node Output - Summary updated]: {state_update['summary'][:150]}...")
                if "current_plan" in state_update and state_update["current_plan"]:
                    print(f"[Node Output - Plan updated]: {state_update['current_plan'][:100]}...")
                if "finished" in state_update:
                    print(f"[Node Output - Finished flag]: {state_update['finished']}")
                
                # Capture the last state update
                final_state = state_update
                
        print("\n" + "="*50)
        print("         AGENT EXECUTION COMPLETED          ")
        print("="*50)
        print("Execution finished successfully.")
        print("="*50 + "\n")

    except KeyboardInterrupt:
        print("\nAgent execution interrupted by user.")
    except Exception as e:
        print(f"\nExecution error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Run the long-running LangGraph agent.")
    parser.add_argument(
        "--task", 
        type=str, 
        help="The task you want the agent to accomplish. If not specified, a default test task will run."
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        help="Override CONTEXT_WINDOW_LIMIT (number of messages before summarization/reset)."
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        help="Override MAX_ITERATIONS (max loops to prevent infinite runs)."
    )

    args = parser.parse_args()

    # Default task if none is provided
    task = args.task
    if not task:
        task = (
            "Create a new python script 'hello_world.py' in the workspace root. "
            "It should print a nice welcome message and current system time. "
            "Run it using the terminal tool to verify it executes without error. "
            "Write a short markdown file 'result_summary.md' detailing the script's output, "
            "then call finish_task to complete your work."
        )

    run_agent(task, args.context_limit, args.max_iter)

if __name__ == "__main__":
    main()
