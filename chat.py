from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content( # This is zero-short prompting, where the model is given a direct question or task without the prior examples
    model = "gemini-2.0-flash",
    contents = "Explain how AI works?",
)

print(response.text)