import config

class TokenTracker:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # Pricing per 1,000,000 tokens (GPT-4o standard pricing: $2.50 input / $10.00 output)
    INPUT_COST_PER_M = 2.50
    OUTPUT_COST_PER_M = 10.00

    @classmethod
    def add(cls, prompt, completion, provider="openai"):
        cls.prompt_tokens += prompt
        cls.completion_tokens += completion
        cls.total_tokens += (prompt + completion)
        
        # Display immediate statistics
        print(f"\n--- [Token Usage info] ---")
        print(f"Current call: {prompt} prompt | {completion} completion | {prompt + completion} total tokens")
        
        if provider == "openai":
            call_cost = (prompt / 1_000_000) * cls.INPUT_COST_PER_M + (completion / 1_000_000) * cls.OUTPUT_COST_PER_M
            total_cost = (cls.prompt_tokens / 1_000_000) * cls.INPUT_COST_PER_M + (cls.completion_tokens / 1_000_000) * cls.OUTPUT_COST_PER_M
            print(f"Cost of call: ${call_cost:.5f} | Total run cost so far: ${total_cost:.5f}")
        else:
            print(f"Running locally (Ollama) - cost is $0.00")
            
        print(f"Total tokens in this run so far: {cls.total_tokens} tokens")
        print("---------------------------\n")

    @classmethod
    def reset(cls):
        cls.prompt_tokens = 0
        cls.completion_tokens = 0
        cls.total_tokens = 0
