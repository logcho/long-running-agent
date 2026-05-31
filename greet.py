from tts_engine import create_synthetic_voice

def deliver_greeting():
    """
    Delivers the personalized greeting using the synthetic voice engine.
    """
    greeting_text = "Hello! I am online, and I am excited to show you my new voice capabilities."
    output_file = "greeting_confirmation.txt"
    create_synthetic_voice(greeting_text, output_file)

if __name__ == "__main__":
    deliver_greeting()