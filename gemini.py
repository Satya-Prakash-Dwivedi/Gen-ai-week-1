import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Setup conversation with the assistant
SYSTEM_PROMPT = """
You are an AI assistant who is expert in breaking down complex problems and resolving user queries.

For a given user input, follow these steps in order:
1. "analyse"
2. "think" (can repeat multiple times)
3. "output"
4. "validate"
5. "result"

Rules:
- Output a single JSON object at each step:
  { "step": "string", "content": "string" }
- Think 5-6 times before giving the final output.
- Wait for the next prompt before continuing to the next step.
- Be very careful with mathematical/logical problems.

Example Input: What is 2 + 2
Example Output:
{ "step": "analyse", "content": "The user is asking a basic arithmetic operation involving addition." }
{ "step": "think", "content": "To solve this, I should add 2 and 2." }
{ "step": "output", "content": "4" }
{ "step": "validate", "content": "Double-checking: 2 + 2 equals 4, so the output is correct." }
{ "step": "result", "content": "2 + 2 = 4, calculated by adding the operands." }

Respond one step at a time.
"""

# Helper to clean and parse JSON from model output
def try_parse_json(text):
    # Strip markdown code block syntax if present
    if text.startswith("```"):
        lines = text.strip().splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Try parsing each line individually if multi-line JSON blocks exist
        for line in text.splitlines():
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue
    return None

# Start chat
chat = genai.GenerativeModel("gemini-2.0-flash").start_chat()
chat.send_message(SYSTEM_PROMPT)

# Get user query
query = input("🧑 Your Question: ")
chat.send_message(f"User Input: {query}")

# Step-by-step conversation loop
while True:
    response = chat.send_message("Next step, please respond in JSON format only.")
    parsed = try_parse_json(response.text)

    if not parsed:
        print("❌ Could not parse valid JSON. Full response:\n", response.text)
        break

    step = parsed.get("step")
    content = parsed.get("content")

    if not step or not content:
        print("⚠️ Incomplete step or content:\n", parsed)
        break

    if step == "result":
        print(f"\n✅ FINAL RESULT: {content}")
        break
    else:
        print(f"🧠 {step.upper()}: {content}")
