from openai import OpenAI

client = OpenAI()

prompt = "What is a cool fact about New Zealand?"

token_limit = 100

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=token_limit
)

# gpt-4o-mini pricing, expressed as dollars per single token
input_token_price = 0.15 / 1_000_000   # $0.15 per 1M input tokens
output_token_price = 0.6 / 1_000_000   # $0.60 per 1M output tokens

# Actual number of input tokens the request used
input_tokens = response.usage.prompt_tokens
# Assume the worst case: the model used every token it was allowed
output_tokens = token_limit

# Estimate the maximum this call could cost
cost = (input_tokens * input_token_price + output_tokens * output_token_price)
print(f"Estimated cost: ${cost:.6f}")