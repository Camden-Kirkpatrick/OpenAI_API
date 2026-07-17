"""
Semantic product search using OpenAI embeddings.

This script embeds each product's short description and a search term into
vectors that capture meaning, then uses cosine distance to find which product
is closest in meaning to the search term. It matches by meaning rather than
keywords, so a search like "computer" can find a smartphone even though that
exact word never appears in the description. The title of the best match is
printed.
"""

from openai import OpenAI
from scipy.spatial import distance
import numpy as np

# Connect to the API (reads the API key from the environment).
client = OpenAI()

# Turn one or more texts into embedding vectors.
def create_embeddings(texts):
  # Send the text(s) to OpenAI and get back the embedding response.
  response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
  )
  # Convert the response object into a plain dictionary we can index into.
  response_dict = response.model_dump()

  # The response holds one entry per input text; pull out just the vectors.
  # Always returns a LIST of vectors, even when given a single string.
  return [data['embedding'] for data in response_dict['data']]

products = [
    {
        "title": "Smartphone X1",
        "short_description": "The latest flagship smartphone with AI-powered features and 5G connectivity.",
        "price": 799.99,
        "category": "Electronics",
        "features": ["6.5-inch OLED display", "128GB storage", "Triple-camera system", "5G support"]
    },
    {
        "title": "Luxury Watch",
        "short_description": "Elegant timepiece with a genuine leather strap and stainless steel case.",
        "price": 249.99,
        "category": "Accessories",
        "features": ["Water-resistant", "Analog display", "Quartz movement", "Leather strap"]
    },
    {
        "title": "Wireless Earbuds",
        "short_description": "Immerse yourself in music with these high-quality wireless earbuds.",
        "price": 129.99,
        "category": "Electronics",
        "features": ["Active noise cancellation", "Bluetooth 5.0", "Long battery life", "Touch controls"]
    },
    {
        "title": "Stainless Steel Cookware Set",
        "short_description": "Upgrade your kitchen with this premium stainless steel cookware set.",
        "price": 149.99,
        "category": "Home & Kitchen",
        "features": ["10-piece set", "Non-stick coating", "Dishwasher safe", "Induction compatible"]
    },
    {
        "title": "Hiking Backpack",
        "short_description": "Explore the outdoors with this durable and spacious hiking backpack.",
        "price": 89.99,
        "category": "Outdoor",
        "features": ["Water-resistant", "Multiple compartments", "Adjustable straps", "40L capacity"]
    }
]

# Collect every product's description into a list of strings.
product_descriptions = [product["short_description"] for product in products]

# Embed all descriptions in a single API call, returning one vector each.
product_embeddings = create_embeddings(product_descriptions)

# Attach each vector back onto its product; index i keeps them matched.
for i, product in enumerate(products):
    product['embedding'] = product_embeddings[i]

# Embed the search text so it can be compared against the products.
# [0] grabs the single vector out of the list that create_embeddings returns.
search_text = "computer"
search_embedding = create_embeddings(search_text)[0]

# Measure how far each product is from the search term in meaning.
distances = []
for product in products:
  # Cosine distance: smaller means more semantically similar.
  dist = distance.cosine(search_embedding, product["embedding"])
  distances.append(dist)

# argmin gives the index of the smallest distance (the closest match).
min_dist_ind = np.argmin(distances)

# Use that index to look up and print the winning product's title.
print(products[min_dist_ind]["title"])