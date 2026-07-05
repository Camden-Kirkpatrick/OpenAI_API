from openai import OpenAI

client = OpenAI()  # automatically reads OPENAI_API_KEY from the environment

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say hello in three languages."}
    ],
    max_completion_tokens=100 # limit token usage
)

print(response.choices[0].message.content)