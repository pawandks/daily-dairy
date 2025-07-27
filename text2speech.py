import os
from gtts import gTTS
from dotenv import load_dotenv
import requests
import google.generativeai as genai
import datetime

# Load .env and Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"❌ Failed to fetch URL content. Status: {response.status_code}")
            return ""
    except Exception as e:
        print(f"❌ Error fetching URL: {e}")
        return ""

def generate_text_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text

def save_text_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Text saved to: {filename}")

def text_to_speech(text, audio_filename):
    tts = gTTS(text)
    tts.save(audio_filename)
    print(f"✅ Audio saved as: {audio_filename}")

def main():
    timestamp = get_timestamp()
    print("💬 Choose input option:")
    print("1️⃣ Type text manually")
    print("2️⃣ Enter URL to fetch content")
    print("3️⃣ Use Gemini to generate text from a prompt")

    choice = input("Enter 1, 2, or 3: ")
    final_text = ""

    if choice == "1":
        final_text = input("✍️ Enter your text: ")
    elif choice == "2":
        url = input("🌐 Enter URL: ")
        final_text = get_text_from_url(url)
    elif choice == "3":
        prompt = input("⚡ Enter your prompt for Gemini: ")
        final_text = generate_text_with_gemini(prompt)
    else:
        print("❌ Invalid choice. Exiting.")
        return

    if not final_text.strip():
        print("❌ No text found or entered. Exiting.")
        return

    # Save original text
    text_filename = f"text_{timestamp}.txt"
    save_text_to_file(final_text, text_filename)

    # Optionally summarize using Gemini
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    summary_response = model.generate_content(f"Summarize this text:\n\n{final_text}")
    summary_text = summary_response.text

    summary_filename = f"summary_{timestamp}.txt"
    save_text_to_file(summary_text, summary_filename)

    # Convert summary text to speech
    audio_filename = f"audio_{timestamp}.mp3"
    text_to_speech(summary_text, audio_filename)

if __name__ == "__main__":
    main()
