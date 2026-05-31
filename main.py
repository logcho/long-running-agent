import sys
import argparse
import json
import config
from src.agent import agent_app
from src.tools.memory_tool import memory_manager
from src.tools.rag_tool import rag_system

def run_agent(task_text, context_limit=None, max_iter=None, provider=None):
    # Override configuration if provided via CLI
    if provider is not None:
        config.LLM_PROVIDER = provider.lower()
        print(f"[Config] Override: LLM_PROVIDER = {config.LLM_PROVIDER}")
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
        current_state = initial_state.copy()
        for step in agent_app.stream(initial_state, {"recursion_limit": config.MAX_ITERATIONS * 3}):
            for node_name, state_update in step.items():
                print(f"\n>>> Node Finished: {node_name}")
                current_state.update(state_update)
                if "summary" in state_update and state_update["summary"]:
                    print(f"[Node Output - Summary updated]: {state_update['summary'][:150]}...")
                if "current_plan" in state_update and state_update["current_plan"]:
                    print(f"[Node Output - Plan updated]: {state_update['current_plan'][:100]}...")
                if "finished" in state_update:
                    print(f"[Node Output - Finished flag]: {state_update['finished']}")
                
        from src.token_tracker import TokenTracker
        print("\n" + "="*50)
        print("         AGENT EXECUTION COMPLETED          ")
        print("="*50)
        print("Execution finished successfully.")
        print("\n--- Final Summary of Accomplished Task ---")
        print(current_state.get("summary", "No summary provided by the agent."))
        print("-" * 42)
        print(f"Total Run Prompt Tokens: {TokenTracker.prompt_tokens}")
        print(f"Total Run Completion Tokens: {TokenTracker.completion_tokens}")
        print(f"Total Run Tokens Used: {TokenTracker.total_tokens}")
        if getattr(config, "LLM_PROVIDER", "ollama").lower() == "openai":
            total_cost = (TokenTracker.prompt_tokens / 1_000_000) * TokenTracker.INPUT_COST_PER_M + (TokenTracker.completion_tokens / 1_000_000) * TokenTracker.OUTPUT_COST_PER_M
            print(f"Estimated OpenAI Cost: ${total_cost:.5f}")
        else:
            print("Estimated API Cost: $0.00 (Ollama)")
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
        help="The task you want the agent to accomplish. If not specified, the CLI will prompt you."
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai"],
        help="Select LLM provider ('ollama' or 'openai')."
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
        print("\n--- Long-Running LangGraph Agent CLI ---")
        try:
            task = input("Enter the task you want the agent to accomplish (leave empty for default test task):\n> ").strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
            
        if not task:
            task = (
                "Create a new python script 'hello_world.py' in the workspace root. "
                "It should print a nice welcome message and current system time. "
                "Run it using the terminal tool to verify it executes without error. "
                "Write a short markdown file 'result_summary.md' detailing the script's output, "
                "then call finish_task to complete your work."
            )
            print(f"Using default test task: {task}\n")

    run_agent(task, args.context_limit, args.max_iter, args.provider)

if __name__ == "__main__":
    main()
