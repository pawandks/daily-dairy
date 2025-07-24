import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please create one and add your key.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash') # You can choose other models like 'gemini-1.5-flash'


def get_website_content(url):
    """Fetches the main textual content from a given URL."""
    try:
        # ✅ Add headers to mimic a browser (fixes 406 error)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text from common content tags, prioritizing readability
        text_content = []
        for tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']:
            for element in soup.find_all(tag):
                text_content.append(element.get_text(separator=' ', strip=True))

        # Join and clean up the text
        full_text = "\n".join(filter(None, text_content))  # Filter out empty strings
        return full_text if full_text else "No substantial text content found on the page."

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred while parsing content: {e}"

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text from common content tags, prioritizing readability
        text_content = []
        for tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']:
            for element in soup.find_all(tag):
                text_content.append(element.get_text(separator=' ', strip=True))

        # Join and clean up the text
        full_text = "\n".join(filter(None, text_content)) # Filter out empty strings
        return full_text if full_text else "No substantial text content found on the page."

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred while parsing content: {e}"

def summarize_text_with_gemini(text, max_length=500):
    """Summarizes the given text using the Gemini API."""
    if len(text) < 50: # Avoid sending very short, unsummarizable text
        return "The provided text is too short to summarize effectively."
    try:
        prompt = f"Summarize the following text concisely and accurately, keeping it under {max_length} words:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error summarizing with Gemini API: {e}"

def save_summary_to_file(url, summary_text):
    """Saves the URL and its summary to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"website_summary_{timestamp}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Website URL: {url}\n")
            f.write(f"Summary Date: {timestamp}\n\n")
            f.write("--- Summary ---\n")
            f.write(summary_text)
            f.write("\n-----------------\n")
        print(f"\nSummary saved to {filename}")
    except IOError as e:
        print(f"Error saving summary to file: {e}")

def main():
    print("Welcome to the Website Summarizer!")
    print("Enter 'exit' at any time to quit.")

    while True:
        website_url = input("\nEnter the website URL to summarize (e.g., https://example.com): ")
        if website_url.lower() == 'exit':
            break

        print(f"Fetching content from: {website_url}...")
        content = get_website_content(website_url)

        if content.startswith("Error"):
            print(content)
            continue
        elif content == "No substantial text content found on the page.":
            print(content)
            save_summary_to_file(website_url, content) # Save even if no content to note
            continue

        print("Content fetched. Summarizing with Gemini API...")
        summary = summarize_text_with_gemini(content)

        print("\n--- Generated Summary ---")
        print(summary)
        print("-------------------------")

        save_summary_to_file(website_url, summary)

if __name__ == "__main__":
    main()