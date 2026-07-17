from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Tell me a couple random facts"}
    ],
    max_completion_tokens=100,
    temperature=2 # temperature determines the amount of randomness from the AI's response (0: deterministic, 1: neutral, 2: random)
)

print(response.choices[0].message.content)