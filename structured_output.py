"""
Structured outputs with the OpenAI Responses API.

Instead of asking the model for JSON in the prompt and hoping it complies, this
hands the model a schema (a Pydantic model) and constrains generation to fit it.
The result is guaranteed to be valid, complete, and correctly typed, so there's
no defensive parsing on our end.

How it works:
  1. Define the shape you want as a Pydantic BaseModel. Each Field's description
     doubles as guidance to the model about what that field means.
  2. Pass it via text_format= to client.responses.parse() (note: parse(), not
     the usual create()).
  3. Read response.output_parsed to get a ready made BandRecommendation object.
     No json.loads, no try/except, just typed attributes like recommendation.band.

This beats requesting JSON in the system role, which the model can ignore, wrap
in prose, or format wrong. Reach for structured outputs whenever the result
feeds other code (a database, app logic, an API) rather than a human reader.
"""

from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

# Define the band recommendation schema
class BandRecommendation(BaseModel):
    band: str = Field(description="Band name")
    genre: str = Field(description="Primary music genre")
    vibe: str = Field(description="A couple words that fit the band")
    why: str = Field(description="One sentence explaining why this matches")

# Generate structured recommendation
response = client.responses.parse(
    model="gpt-5.4-mini",
    instructions="You are a knowledgeable band recommender",
    input="Recommend a band for someone who loves Rush",
    text_format=BandRecommendation,
)

# Extract the parsed output and results
recommendation = response.output_parsed
print(f"Band: {recommendation.band}")
print(f"Genre: {recommendation.genre}")
print(f"Vibe: {recommendation.vibe}")
print(f"Reason: {recommendation.why}")