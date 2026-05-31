import os
from google.cloud import texttospeech

# --- Configuration and Setup ---
# The client automatically handles authentication using GOOGLE_APPLICATION_CREDENTIALS
try:
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    print(f"Error initializing TTS Client: {e}")
    print("Please ensure Google Cloud credentials are set using 'gcloud auth application-default login'.")
    exit()

def synthesize_speech(text: str, filename: str) -> bool:
    """
    Synthesizes speech from text and saves it to a local file (MP3).

    Args:
        text: The text string to be converted to speech.
        filename: The name of the output MP3 file.

    Returns:
        True if synthesis was successful, False otherwise.
    """
    print(f"\n--- 🎤 Synthesizing audio for '{filename}' ---")
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Select a voice (e.g., 'en-US-Standard-C')
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", 
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Select the type of audio file to return
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    try:
        # Perform the TTS request
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # The response's audio_content is binary. Write it to the file.
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            print(f"✅ Success: Audio content written to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error during TTS synthesis: {e}")
        return False

def run_conversation_simulation(user_text: str, model_text: str):
    """
    Orchestrates the TTS process for a simulated two-way conversation.
    
    Args:
        user_text: The text the 'user' spoke.
        model_text: The text the 'model' spoke.
    """
    print("\n" + "="*50)
    print("🎙️ STARTING CONVERSATION SIMULATION 🎙️")
    print("="*50)

    # 1. Simulate User Input
    if synthesize_speech(user_text, "user_input.mp3"):
        print("\n[SYSTEM]: User turn audio successfully generated and saved.")
    else:
        print("\n[SYSTEM]: Failed to generate user audio. Aborting simulation.")
        return

    # 2. Simulate Model Response
    if synthesize_speech(model_text, "model_response.mp3"):
        print("\n[SYSTEM]: Model turn audio successfully generated and saved.")
    else:
        print("\n[SYSTEM]: Failed to generate model audio. Simulation finished with partial success.")

    print("\n" + "="*50)
    print("🎉 CONVERSATION SIMULATION COMPLETE 🎉")
    print("The audio files 'user_input.mp3' and 'model_response.mp3' should be available.")
    print("="*50)

# --- Execution Block ---
if __name__ == "__main__":
    # Example conversation texts for simulation
    user_input = "Hello! I am excited to work on my conversational agent. Can you help me with the voice component?"
    model_response = "Of course! Creating an artificial voice is a highly achievable and rewarding project. We can use the Google Cloud Text-to-Speech API for this."

    run_conversation_simulation(user_input, model_response)