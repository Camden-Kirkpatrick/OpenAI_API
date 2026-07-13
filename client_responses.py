# ─────────────────────────────────────────────────────────────
# OpenAI Responses API — the newer, recommended interface
# ─────────────────────────────────────────────────────────────
# Uses client.responses.create() instead of the older
# client.chat.completions.create().
#
# Key differences from Chat Completions:
#   • input=        -> plain string prompt (replaces the messages=[] list of dicts)
#   • instructions= -> system-style guidance (replaces the "system" role message)
#   • max_output_tokens=  -> renamed from max_completion_tokens
#
# Reading the result:
#   • response.output_text        -> convenience: the full text answer
#   • response.output             -> structured list of output items (verbose)
#   • response.usage.output_tokens -> token count for the generated output
#
# Setup: client = OpenAI() reads OPENAI_API_KEY from the environment.
# ─────────────────────────────────────────────────────────────

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