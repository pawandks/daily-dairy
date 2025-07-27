import os
import speech_recognition as sr
from gtts import gTTS
import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def record_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Please speak now...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    print("✅ Recording complete!")
    return audio

def transcribe_audio(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        print(f"📝 Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"❌ API request error: {e}")
        return ""

def modify_text_with_gemini(text, user_prompt):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(f"{user_prompt}\n\n{text}")
    return response.text

def save_text(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Text saved to: {filename}")

def convert_text_to_speech(text, audio_filename):
    tts = gTTS(text)
    tts.save(audio_filename)
    print(f"✅ Output audio saved: {audio_filename}")

def main():
    timestamp = get_timestamp()

    # Filenames
    text_filename = f"transcription_{timestamp}.txt"
    modified_text_filename = f"modified_text_{timestamp}.txt"
    audio_output_filename = f"output_audio_{timestamp}.mp3"

    # Step 1: Record speech
    audio = record_speech()

    # Step 2: Transcribe
    text = transcribe_audio(audio)
    if not text:
        print("⚠️ No text to process. Exiting.")
        return

    save_text(text, text_filename)

    # Step 3: Use Gemini to modify text (prompt from you)
    choice = input("💬 Do you want to modify text using Gemini (summarize/rephrase/translate)? (yes/no): ").strip().lower()

    if choice == "yes":
        user_prompt = input("✍️ Enter your custom prompt for Gemini (e.g., 'Summarize this', 'Translate to Hindi'): ")
        modified_text = modify_text_with_gemini(text, user_prompt)
        print(f"✨ Modified text: {modified_text}")

        save_text(modified_text, modified_text_filename)
        final_text = modified_text
    else:
        final_text = text

    # Step 4: Convert final text to speech
    convert_text_to_speech(final_text, audio_output_filename)

if __name__ == "__main__":
    main()
