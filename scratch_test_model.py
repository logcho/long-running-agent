import sys
from ollama import Client

def main():
    print("Testing connection to Ollama...")
    try:
        client = Client(host="http://localhost:11434")
        response = client.chat(
            model="gemma4:latest",
            messages=[{"role": "user", "content": "Say hello!"}]
        )
        print("Response received:")
        print(response['message']['content'])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
