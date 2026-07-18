"""
Classify restaurant reviews by sentiment using embeddings.

Each sentiment label (Positive, Neutral, Negative) and each review is turned
into an embedding vector with OpenAI. A review is then matched to the sentiment
whose vector is closest by cosine distance, and the resulting label is printed.
"""

from openai import OpenAI
from scipy.spatial import distance

client = OpenAI()

# The candidate sentiment classes, each with a short description.
# The descriptions give the embedding model extra context to compare against.
sentiments = [
    {'label': 'Positive', 'description': 'A positive restaurant review'},
    {'label': 'Neutral', 'description':'A neutral restaurant review'},
    {'label': 'Negative', 'description': 'A negative restaurant review'}
]

# The reviews we want to classify.
reviews = [
    "The food was delicious!",
    "The service was a bit slow but the food was good",
    "The food was cold, really disappointing!"
]

# Turn one or more texts into embedding vectors.
def create_embeddings(texts):
  # Send the text(s) to OpenAI and get back the embedding response.
  response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
  )
  # Convert the response object into a plain dictionary we can index into.
  response_dict = response.model_dump()

  # Pull out just the vectors; always returns a LIST, even for a single string.
  return [data['embedding'] for data in response_dict['data']]


# Flatten a sentiment dict into one text blob so all its fields get embedded.
def create_sentiment_text(sentiment):
  # Combine the label and description into a single string for embedding.
  return (
        f"""Sentiment: {sentiment['label']}
        Description: {sentiment['description']}"""
    )

# Build the text blobs for every sentiment, then embed them all at once.
sentiment_text = [create_sentiment_text(sentiment) for sentiment in sentiments]
sentiment_embeddings = create_embeddings(sentiment_text)

# Embed every review in a single call.
review_embeddings = create_embeddings(reviews)

# Find the embedding in `embeddings` that is closest to `query_vector`.
def find_closest(query_vector, embeddings):
  distances = []
  # Measure the cosine distance from the query to each candidate embedding.
  for index, embedding in enumerate(embeddings):
    dist = distance.cosine(query_vector, embedding)
    # Keep the distance alongside its index so we can look it up later.
    distances.append({"distance": dist, "index": index})
  # The smallest distance means the most similar vector.
  return min(distances, key=lambda x: x["distance"])

# For each review, find its nearest sentiment and print the predicted label.
for index, review in enumerate(reviews):
  # Compare this review's embedding against every sentiment embedding.
  closest = find_closest(review_embeddings[index], sentiment_embeddings)
  # Translate the winning index back into its human readable label.
  label = sentiments[closest['index']]['label']
  print(f'"{review}" was classified as {label}')
