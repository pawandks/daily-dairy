import requests

def generate_image(prompt):
    print(f"Generating image for prompt: {prompt}")

    API_URL = "https://hogiahien-counterfeit-v30-edited.hf.space/run/predict"
    payload = {
        "data": [
            prompt,  # prompt
            "",      # negative prompt
            20,      # steps
            7.5,     # guidance scale
            512,     # width
            512      # height
        ]
    }

    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        output_url = response.json()["data"][0]
        img_data = requests.get(output_url).content
        with open("output.png", "wb") as f:
            f.write(img_data)
        print("✅ Image saved as output.png")
    else:
        print("❌ Error:", response.status_code, response.text)

if __name__ == "__main__":
    prompt = input("Enter a prompt to generate image: ")
    generate_image(prompt)
