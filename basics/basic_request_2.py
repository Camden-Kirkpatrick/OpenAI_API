from openai import OpenAI

client = OpenAI()

country = input("Enter a country, to see its capital: ")

prompt = f"""
    What is the capital of {country}?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=100
)

print(response.choices[0].message.content)