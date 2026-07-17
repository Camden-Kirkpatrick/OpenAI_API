"""
Streaming a response from the OpenAI Responses API.

Instead of waiting for the model to finish generating before showing any output,
this opens a live connection and receives the response as a series of typed
events. Text arrives in small chunks ("deltas") that are printed as they come in,
producing a real-time typing effect. Each chunk is also accumulated into
`current_text` so the complete response is available once streaming ends.

Events handled:
  - response.created         -> the model has begun generating
  - response.output_text.delta -> a new chunk of text (printed immediately)
  - response.completed       -> generation is fully done; print the full text
"""

from openai import OpenAI

client = OpenAI()

prompt = "Pick a random dish, and list the ingridients needs to make it."

# Open a connection for a streaming request
with client.responses.create(model="gpt-5.4-mini", input=prompt, stream=True) as stream:
    current_text = ""

    # Complete the output text streaming
    for event in stream:
        if event.type == "response.created":
            print("Response started...\n")
        # Print the previous text, plus the new chunk
        # elif event.type == "response.output_text.delta":
        #     current_text += event.delta
        #     print(current_text)
        # Print only the new chunk of data
        elif event.type == "response.output_text.delta":
            current_text += event.delta
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            print(f"\n\nFull response:\n{current_text}")