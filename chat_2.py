import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

system_instruction = """
You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query, help the user to solve it along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 by 10. Fun fact: you can even multiply 10 * 3 which gives the same result.

Input: Why is sky blue?
Output: Bruh? You alright? Is it a maths query?
"""

chat = genai.GenerativeModel("gemini-2.0-flash").start_chat()

user_input = "what is a mobile phone?"

full_prompt = f"{system_instruction}\n\nUser: {user_input}"

response = chat.send_message(full_prompt)

# Output response
print(response.text)
