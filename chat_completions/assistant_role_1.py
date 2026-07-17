"""
The assistant message here is written by us (not model output) to show
the desired response format - the model imitates it for the next question.
This is one-shot prompting: pairing an example user request with a sample
assistant reply teaches the format by example rather than by instruction.
"""

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    # Add a user and assistant message for in-context learning
    messages=[
        {"role": "system", "content": "You are a helpful Geography tutor that generates concise summaries for different countries."},
        {"role": "user", "content": "Give me a quick summary of Portugal."},
        {"role": "assistant", "content": "Portugal is a country in Europe that borders Spain. The capital city is Lisboa."},
        {"role": "user", "content": "Give me a quick summary of Greece."}
    ]
)

print(response.choices[0].message.content)