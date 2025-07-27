import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json

def get_all_links(url):
    internal_links = set()
    external_links = set()

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Cannot access {url}: {e}")
        return internal_links, external_links

    soup = BeautifulSoup(response.text, 'html.parser')
    domain_name = urlparse(url).netloc

    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        href = urljoin(url, href)  # Convert relative to absolute URL

        if urlparse(href).scheme in ['http', 'https']:
            if domain_name in urlparse(href).netloc:
                internal_links.add(href)
            else:
                external_links.add(href)

    return internal_links, external_links

def save_to_json(input_url, internal_links, external_links):
    data = {
        "input_url": input_url,
        "total_internal_links": len(internal_links),
        "total_external_links": len(external_links),
        "internal_links": sorted(list(internal_links)),
        "external_links": sorted(list(external_links))
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("\n✅ Output saved to 'result.json' in your local folder.")

def main():
    website_url = input("Enter the website URL (e.g. https://example.com): ").strip()

    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url

    internal, external = get_all_links(website_url)

    print(f"\n🔗 Found {len(internal)} internal links and {len(external)} external links.")
    save_to_json(website_url, internal, external)

if __name__ == "__main__":
    main()
