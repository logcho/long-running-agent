import os
import sys

# The secret word is retrieved from the persistent memory/environment.
# In a real scenario, we might read it from an environment variable or a dedicated config file.
# For this exercise, we hardcode the confirmed value based on the prior steps.
SECRET_WORD = "ANTIGRAVITY-RAG-ACTIVE"

def print_secret_word():
    """Prints the stored secret word to standard output."""
    print("--- Greeting Script ---")
    print(f"Secret Word Found: {SECRET_WORD}")
    print("-----------------------")

if __name__ == "__main__":
    print_secret_word()