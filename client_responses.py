from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4-mini",
    instructions="Be concise",
    input="Explain LLMS in terms an 8 year old can understand",
    max_output_tokens=100
)

print(response.output)
print(response.output_text)
print("Output tokens used:", response.usage.output_tokens)