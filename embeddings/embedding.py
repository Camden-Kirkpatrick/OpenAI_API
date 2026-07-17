"""
Embeddings turn text into a fixed-length list of numbers (a vector) that
captures its meaning. Text with similar meaning maps to vectors that are
close together, so the distance between two vectors becomes a measure of how
semantically related the underlying pieces of text are.

This is useful because it lets programs work with meaning instead of just
matching exact words. Common uses include:
  - Semantic search: find results by meaning, not just keyword overlap
  - Clustering / classification: group or label text by topic
  - Recommendations: surface items similar to what a user liked
  - RAG (retrieval-augmented generation): fetch relevant context to feed an LLM
  - Deduplication: detect near-duplicate or paraphrased text

The model below returns one such vector for the given input, which you would
typically store (e.g. in a vector database) and later compare against other
embeddings using a similarity metric like cosine similarity.
"""

from openai import OpenAI

client = OpenAI()

# Create a request to obtain embeddings
response = client.embeddings.create(
  model="text-embedding-3-small",
  input="This can contain any text."
)

# Convert the response into a dictionary
response_dict = response.model_dump()
# Print the embedding and token data
print("Embedding:", response_dict["data"][0]["embedding"])
print("Tokens used:", response_dict["usage"]["total_tokens"])