# The system role is used to set the behavior, instructions, and persona of the assistant before the conversation begins.

from openai import OpenAI

client = OpenAI()

sys_msg = """You are a study planning assistant that creates plans for learning new skills.

If these skills are non related to languages, return the message:

'Apologies, only learning plans associated with language are allowed.'
"""

response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": sys_msg}, # Give instructions for how the AI should behave
        {"role": "user", "content": "I need help learning Russian"},
    ],
    max_completion_tokens=100
)

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": "I need help learning Calculus"},
    ],
    max_completion_tokens=100
)


print(response1.choices[0].message.content)
print()
# This returns the error, since Calculas is not language realted
print(response2.choices[0].message.content)