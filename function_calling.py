"""
Function calling (tools) with the OpenAI Responses API.

A model can't run code or reach live data on its own. It's like a chef who
can't leave the kitchen: it can decide and talk, but YOU (your code) are the
waiter who actually fetches things. Function calling is that handshake:

  1. You describe your functions to the model via `tools=` (name, description,
     and a schema of the arguments). The model never sees the code inside, only
     this description, so it knows the function exists and when to ask for it.
  2. Given a prompt, the model may decide it needs one. It does NOT run it. It
     names the function it wants and the arguments to use. With many tools, the
     model tells you WHICH one by name; you just match that name to your function.
  3. YOUR code runs the real function and sends the result back, matched to the
     request by `call_id`.
  4. The model reads that result and writes the final natural-language answer.

So it's two API calls with your code in the middle: the model asks, you answer,
the model responds. (Steps 2 to 4 may repeat if the model wants several tools.)

`get_weather` below returns canned data so the script runs without an external
API. In real code this is where you'd call a live service, query a DB, etc.
"""

import json
from openai import OpenAI

client = OpenAI()


# Step 0: the actual function the model can ask us to run.
# YOU write this, a normal Python function. The model only knows its name.
def get_weather(city):
    fake_data = {
        "Tokyo": "22°C, clear",
        "London": "14°C, rainy",
        "Cairo": "35°C, sunny",
    }
    return fake_data.get(city, "Unknown city")


# Step 1: describe the function to the model.
# `parameters` is a JSON Schema describing the arguments. "type": "object" means
# the arguments come as a bundle of named fields (i.e. {"city": ...}); inside,
# each field gets its own type ("city" is a string). "required" lists the fields
# the model must provide.
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Tokyo'",
                }
            },
            "required": ["city"],
        },
    }
]

# A running transcript of the conversation. We append the model's requests and
# our results so the model has the full context on the second call.
input_list = [{"role": "user", "content": "What's the weather in Tokyo?"}]

# Step 2: first call. The model decides whether to use a tool.
response = client.responses.create(
    model="gpt-5.4-mini",
    tools=tools,
    input=input_list,
)

# Staple the model's output (which includes any function_call requests) onto
# the transcript so it can see its own request on the second call.
input_list += response.output

# Step 3: run any function the model asked for, and return the result.
for item in response.output:
    if item.type == "function_call":
        # The model produces text, so arguments arrive as a JSON *string*
        # like '{"city": "Tokyo"}'. json.loads rebuilds it into a real dict.
        args = json.loads(item.arguments)
        print(f"Model requested: {item.name}({args})")

        # This is the moment the function actually runs. Here there's only one
        # function, so we call it directly:
        result = get_weather(**args)

        # With MULTIPLE functions you'd match item.name to the right one. The
        # model tells you which by name:
        #
        #   if item.name == "get_weather":
        #       result = get_weather(**args)
        #   elif item.name == "get_stock_price":
        #       result = get_stock_price(**args)
        #   elif item.name == "send_email":
        #       result = send_email(**args)

        # Write the result back, tagged with the same call_id so the model
        # knows which request this answers.
        input_list.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": str(result),
        })

# Step 4: second call. The model turns the tool result into an answer.
# The transcript now holds the question, the model's request, AND our result,
# so the model can finally write a normal sentence.
final = client.responses.create(
    model="gpt-5.4-mini",
    tools=tools,
    input=input_list,
)

print(final.output_text)
