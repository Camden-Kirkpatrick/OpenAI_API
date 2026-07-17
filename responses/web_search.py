"""
Web search with the OpenAI Responses API.

Sends a query to an OpenAI model with the built-in `web_search` tool enabled,
letting the model search the live web to answer. The request opts into
`web_search_call.action.sources` so the response includes the URLs the model
consulted.

This is useful because a model on its own can only draw on information from up
to its training cutoff - anything newer is invisible to it. Enabling web search
lifts that limitation, letting the model pull in current information (like the
latest Python version) at query time.

The script then walks `response.output`, prints the source URL of every
`web_search_call` item, and finally prints the model's synthesized answer
(`response.output_text`).
"""

from openai import OpenAI

client = OpenAI()

# Create a response with web search enabled
response = client.responses.create(
    model="gpt-5.4-mini",
    tools=[{"type": "web_search"}],
    input="What is the latest Python version?",
    include=["web_search_call.action.sources"]
)

print("Sources used in web search:")
for item in response.output:
    if item.type == "web_search_call":
        for source in item.action.sources:
            print(source.url)

print(response.output_text)