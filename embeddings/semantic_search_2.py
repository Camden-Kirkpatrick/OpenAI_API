"""
Semantic product search that returns the top N closest matches.

This script builds a rich text blob for each product (title, description, price,
category, and features), embeds each one into a vector that captures meaning, and
embeds the search query the same way. It then ranks products by cosine distance to
the query and prints the titles of the N closest matches. Because it matches on
meaning rather than keywords, a query like "technology" can surface electronics
even when that exact word never appears in the product text.

Improvements over the earlier single-result version:
  1. Richer input. It embeds create_product_text() (title, description, price,
     category, AND features) instead of only the short_description, so the vector
     captures far more about each product and matches are more accurate.
  2. Ranked top N results, not just one. find_n_closest() sorts every product by
     distance and returns the n best, so you see a ranked shortlist instead of the
     single argmin winner, which is what a real search results page needs.
  3. Distance is paired with its index. Each result keeps {distance, index}, so
     after sorting we can map any result straight back to its product. Sorting
     (rather than a one-shot np.argmin) also makes returning more results trivial.
"""

from openai import OpenAI
from scipy.spatial import distance

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

  # Pull out just the vectors; always returns a LIST, even for a single string.
  return [data['embedding'] for data in response_dict['data']]

# Flatten a product dict into one text blob so all its fields get embedded.
def create_product_text(product):
  return (
        f"""Title: {product['title']}
        Description: {product['short_description']}
        Price: {product['price']}
        Category: {product['category']}
        Features: {', '.join(product['features'])}"""
    )

# Return the n embeddings closest to query_vector, each as {distance, index}.
def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        # Calculate the cosine distance between the query vector and embedding.
        dist = distance.cosine(query_vector, embedding)
        # Keep both the distance and the index so we can find the product later.
        distances.append({"distance": dist, "index": index})
    # Sort so the smallest distances (closest matches) come first.
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    # Return only the top n closest.
    return distances_sorted[0:n]

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

# Build the text blob for every product, then embed them all in one call.
product_texts = [create_product_text(product) for product in products]
product_embeddings = create_embeddings(product_texts)

# Embed the search query; [0] grabs the single vector out of the returned list.
query_text = "technology"
query_vector = create_embeddings(query_text)[0]

# Find the two products closest in meaning to the query.
hits = find_n_closest(query_vector, product_embeddings, 2)

# Print the title of each matching product, closest first.
print(f"Search results for '{query_text}'")
for hit in hits:
  # hit['index'] points back to the product that produced this embedding.
  product = products[hit["index"]]
  print(product["title"])