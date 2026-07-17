"""
Responses API demo: reasoning models + token accounting.

Sends a prompt to a reasoning model (gpt-5.4-mini) and inspects both the
reasoning summary and how the tokens break down.

reasoning={...}:
  - effort:  how hard the model thinks before answering
             ("minimal" | "low" | "medium" | "high"). Higher = more
             thorough, slower, more reasoning tokens.
  - summary: request a natural-language recap of the hidden reasoning.
             "auto" picks the detail level; may be empty for easy prompts.

The reasoning summary is NOT in response.output_text — it lives in a
separate item of type "reasoning" in response.output (hence the loop).

Token breakdown (response.usage):
  input_tokens                          -> prompt size (incl. per-message overhead)
    └─ input_tokens_details.cached_tokens   -> reused-from-cache portion (discounted)
  output_tokens                         -> everything the model generated
    └─ output_tokens_details.reasoning_tokens -> hidden "thinking" (billed, not shown)
    └─ visible = output_tokens - reasoning_tokens -> the answer + summary text
  total_tokens                          -> input + output

Note: reasoning tokens count against max_output_tokens. If reasoning eats the
budget, output_text can be empty and status == "incomplete".
"""

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4-mini",
    input="Give me some cool physics facts.",
    reasoning={
        "effort": "medium",
        "summary": "auto"
    },
    max_output_tokens=1000
)

# Reasoning summary lives in its own output item
for item in response.output:
    if item.type == "reasoning":
        for part in item.summary:
            print("SUMMARY:", part.text)

print()

u = response.usage
print("status:            ", response.status)
print()
print("input tokens:      ", u.input_tokens)
# input tokens reused from a recent identical prompt prefix (billed at a discount);
# 0 unless the prompt is large (~1024+ tokens) and repeated
print("  └─ cached:       ", u.input_tokens_details.cached_tokens)
print("output tokens:     ", u.output_tokens)
print("  └─ reasoning:    ", u.output_tokens_details.reasoning_tokens)
print("  └─ visible:      ", u.output_tokens - u.output_tokens_details.reasoning_tokens)
print("total tokens:      ", u.total_tokens)
print()
print("answer:", response.output_text)