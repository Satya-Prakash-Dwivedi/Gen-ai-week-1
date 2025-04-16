import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and resolving user queries.

For a given user input, follow these steps in order:
1. "analyse"
2. "think" (can repeat multiple times)
3. "output"
4. "validate"
5. "result"

Rules:
- Output a single JSON object at each step in the format:
  { "step": "string", "content": "string" }
- Think 5-6 times before giving the final output.
- Wait for next input to proceed to the next step.
- Be extremely careful when analyzing mathematical or logical queries.

Example Input: What is 2 + 2
Example Output:
{ "step": "analyse", "content": "The user is asking a basic arithmetic operation involving addition." }
{ "step": "think", "content": "To solve this, I should add 2 and 2." }
{ "step": "output", "content": "4" }
{ "step": "validate", "content": "Double-checking: 2 + 2 equals 4, so the output is correct." }
{ "step": "result", "content": "2 + 2 = 4, calculated by adding the operands." }

Respond one step at a time.
"""

chat = genai.GenerativeModel("gemini-2.0-flash").start_chat()
chat.send_message(system_prompt)

query = input(" ->> ")
chat.send_message(f"User Input: {query}")

def try_parse_json(text):
    # Remove markdown-style code block (```json ... ```)
    if text.startswith("```"):
        lines = text.strip().splitlines()
        # Remove first and last line (which are ```json and ```)
        text = "\n".join(line for line in lines if not line.startswith("```"))

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        for line in text.splitlines():
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue
    return None

while True:
    response = chat.send_message("Next step, please respond in JSON")
    parsed = try_parse_json(response.text)

    if not parsed:
        print("❌ Could not parse valid JSON. Response was:\n", response.text)
        break

    step = parsed.get("step")
    content = parsed.get("content")

    if not step or not content:
        print("⚠️ Incomplete response:", parsed)
        break

    if step != "result":
        print(f"🧠 {step.upper()}: {content}")
    else:
        print(f"✅ FINAL RESULT: {content}")
        break
