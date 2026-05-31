from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    State representing the agent's memory, plan, messages, and progress.
    Since we don't use a custom LangGraph reducer for 'messages', any node returning
    'messages' will overwrite the field. This gives us complete control over 
    pruning/resetting the context window during summarization.
    """
    # Active context window messages
    messages: List[Dict[str, Any]]
    
    # Task/Goal description
    task: str
    
    # Combined summary of previous context windows (episodic memory)
    summary: str
    
    # Rules, tricks, or insights the agent has discovered during self-improvement
    lessons_learned: List[str]
    
    # Key-value factual memory (long-term/short-term key details)
    working_memory: Dict[str, Any]
    
    # High-level checklist/plan the agent is executing
    current_plan: str
    
    # Historical description of steps and tools executed
    steps_taken: List[str]
    
    # Count of steps executed to prevent infinite loops
    iterations: int
    
    # Whether the goal is complete
    finished: bool
