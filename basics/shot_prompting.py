"""
Shot-based prompting is a prompt engineering technique where
you include examples - or "shots" within your prompt to teach an AI model
how to format, tone, or structure its response
"""

from openai import OpenAI

client = OpenAI()

prompt = """Classify sentiment as 1-5 (negative to positive):
1. Comfortable, but not very pretty = 2
2. Love these! = 5
3. Unbelievably good! = 
4. Shoes fell apart on the second use. = 
5. The shoes look nice, but they aren't very comfortable. = 
6. Can't wait to show them off! = 
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=100
)

print(response.choices[0].message.content)