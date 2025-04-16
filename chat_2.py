from dotenv import load_dotenv
from google import genai
import os
from google.genai.types import GenerateContentConfig

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

system_prompt = """

    You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multipling 3 by 10. Funfact you can even multiply 10 * 3 which gives same result.

Input: Why is sky blue?
Output: Bruh? You alright? Is it maths query?

"""

response = client.models.generate_content( # This is a few shot prompting technique, where we give few examples before asking it to generate.
    model="gemini-2.0-flash",
    contents="What is agi and how it will impact humanity",
    config=GenerateContentConfig(
        system_instruction= system_prompt
    ),
)

print(response.text)