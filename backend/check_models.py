import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY is not set in environment or .env file.")
else:
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        models = [m["id"] for m in response.json().get("data", [])]
        print("Available Groq Models on your key:")
        for m in models:
            print(f" - {m}")
    else:
        print(f"Failed to retrieve models ({response.status_code}): {response.text}")