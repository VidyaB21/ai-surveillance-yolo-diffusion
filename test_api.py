import requests

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

HEADERS = {
    "Authorization": "Bearer YOUR_HF_TOKEN",
    "Accept": "image/png"
}

payload = {
    "inputs": "A cat sitting on a chair, realistic"
}

# 🔍 DEBUG BEFORE REQUEST
print("Sending request to:", API_URL)

response = requests.post(API_URL, headers=HEADERS, json=payload)

# 🔍 DEBUG AFTER REQUEST
print("Status Code:", response.status_code)
print("Response headers:", response.headers)

if response.status_code == 200:
    with open("test.png", "wb") as f:
        f.write(response.content)
    print("✅ Image generated!")
else:
    print("❌ Error:")
    print(response.text)