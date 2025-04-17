import json, os, requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def query_db(sql):
    pass

def run_command(command):
    result = os.system(command=command)
    return result

def get_weather(city: str):
    print("Tool called : get_weather", city)

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"

available_tools = {
    "get_weather" : {
        "fn" : get_weather,
        "description" : "Takes a city name as an input and return the current weather for the city"
    },
    "run_command": {
        "fn" : run_command,
        "description": "Takes a command as input to execute on system and returns output"
    }
}

system_prompt = f"""

You are an helpful AI Assistant who is specialized in resolving user query. 
You work on start, plan, action, observer mode. 
For the given user query and available tools, plan the step by step execution, based on the planning, 
select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
Wait for the observation and based on the observation from the tool acall resolve the user query.

Rules:
- Follow the output JSON Format.
- Always perfrom one step at a time and wait for next input
- Carefully analyse the user query
+ - Always respond strictly in valid JSON format for every reply, even if you're waiting or unsure
+ - Never say "waiting for the tool" — assume the observation is already handled
Output JSON Format:
{{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function",
}}

Available Tools:
- get_weather : Takes a city name as an input and returns the current weather for the city
- run_command : Takes a command as input to execute on system and returns output

Example:
User Query: What is the weather of new york?
Output: {{"step": "plan", "content": "The user is interested in weather data of new york"}}
Output : {{"step": "plan", "content": "From the available tools I should call get_weather}}
Output: {{"step": "action", "function": "get_weather", "input": "new york"}}
Output: {{"step": "observe", "output": "12 Degree celcius}}
Output: {{"step": "output", "content": "The weather for new york seems to be 12 degrees}}

"""

chat = genai.GenerativeModel("gemini-2.0-flash").start_chat()
chat.send_message(system_prompt)

def try_parse_json(text):
    try:
        if text.startswith("```"):
            lines = text.strip().splitlines()
            text = "\n".join(line for line in lines if not line.startswith("```"))
        return json.loads(text.strip())
    except json.JSONDecodeError:
        for line in text.splitlines():
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue
    return None

    
while True:
    user_query = input('>>')
    if not user_query:
        continue

    chat.send_message(f"User query: {user_query}")

    while True:
        try:
            response = chat.send_message("Next step")
            parsed = try_parse_json(response.text)

            if not parsed:
                print("❌ Could not parse valid JSON. Response was:\n", response.text)
                break

            step = parsed.get("step")
            content = parsed.get("content")
            function = parsed.get("function")
            tool_input = parsed.get("input")

            if step == "plan":
                print(f"🧠 PLAN: {content}")
                continue
            elif step == "action" and function in available_tools:
                print(f"⚙️ ACTION: Calling {function} with input: {tool_input}")
                output = available_tools[function]["fn"](tool_input)
                chat.send_message(json.dumps({
                    "step" : "observe",
                    "content" : output
                }))
                continue

            elif step == "observe":
                print(f"👀 OBSERVED: {content}")
                continue

            elif step == "output":
                print(f"✅ FINAL ANSWER: {content}")
                break

            else:
                print(f"⚠️ Unknown step or error: {parsed}")
                break
        except Exception as e:
            print(f"⚠️ Gemini error {e}")
            break