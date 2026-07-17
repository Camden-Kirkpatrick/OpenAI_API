"""
Interactive chat loop with multi-turn memory.

This is an improved, and different, take on ai_tourist.py:
  - Real back-and-forth: each question is sent and answered immediately,
    instead of collecting every question up front and answering them all at
    the end.
  - Conversation history is kept server-side via the Responses API. We only
    store the previous response's id and pass it as previous_response_id, so
    there's no need to build and resend a growing messages list by hand (as
    ai_tourist.py does with its Chat Completions conversation array).
"""

from openai import OpenAI

client = OpenAI()

sys_prompt = "You are a helpful teacher who provides concise explanations."
latest_response_id = None

while True:
    user_input = input("You: ").strip()
    if (user_input.lower() == "q"):
        break

    response = client.responses.create(
        model="gpt-5.4-mini",
        instructions=sys_prompt,
        input=user_input,
        previous_response_id=latest_response_id
    )

    latest_response_id = response.id
    print(f"\nAssistant: {response.output_text}\n")