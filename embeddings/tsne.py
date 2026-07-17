"""
Visualize product text embeddings in 2D using t-SNE.

This script takes a list of products, creates a text embedding for each
product's short description using OpenAI's text-embedding-3-small model,
then uses t-SNE to reduce those high-dimensional vectors down to two
dimensions so they can be plotted. Each point is labeled with its product
category, so visually similar (semantically related) products cluster
together on the scatter plot.

We reduce to 2 dimensions because a scatter plot has exactly two axes
(x and y), so each product needs exactly two numbers to be drawn as a dot.
The raw embeddings live in ~1536 dimensions, which is where the meaning is
captured but is impossible to view directly; 2D is what our eyes and a flat
plot can actually show, and t-SNE preserves the closeness between similar
products as it shrinks the vectors down.
"""

from openai import OpenAI
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

client = OpenAI()

products = [
    {"title": "Smartphone X1",   "short_description": "The latest flagship smartphone with AI features and 5G.",        "category": "Electronics"},
    {"title": "Wireless Earbuds","short_description": "High-quality wireless earbuds with noise cancellation.",         "category": "Electronics"},
    {"title": "4K Smart TV",     "short_description": "Ultra HD smart television with streaming apps built in.",        "category": "Electronics"},
    {"title": "Luxury Watch",    "short_description": "Elegant timepiece with leather strap and steel case.",           "category": "Accessories"},
    {"title": "Leather Wallet",  "short_description": "Slim genuine leather wallet with card slots.",                   "category": "Accessories"},
    {"title": "Cookware Set",    "short_description": "Premium stainless steel non-stick cookware set.",                "category": "Home & Kitchen"},
    {"title": "Coffee Maker",    "short_description": "Programmable drip coffee maker with thermal carafe.",            "category": "Home & Kitchen"},
    {"title": "Hiking Backpack", "short_description": "Durable water-resistant backpack for outdoor hikes.",            "category": "Outdoor"},
    {"title": "Camping Tent",    "short_description": "Lightweight two-person tent for camping trips.",                 "category": "Outdoor"},
]

# Step 1: Create the embeddings
# Pull out just the descriptions into a plain list of strings.
product_descriptions = [product["short_description"] for product in products]

# Send all descriptions in ONE API call. The model returns a vector
# (list of ~1536 numbers) for each description, in the same order.
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=product_descriptions
)
response_dict = response.model_dump()   # Convert the response object to a plain dict

# Attach each returned vector back onto its product under the 'embedding' key.
# enumerate gives us the index i so we can match product to response by position.
for i, product in enumerate(products):
    product["embedding"] = response_dict["data"][i]["embedding"]

# Step 2: Pull categories and embeddings into parallel lists
# Index i in both lists refers to the SAME product.
categories  = [product["category"]  for product in products]
embeddings  = [product["embedding"] for product in products]

# Step 3: Reduce 1536-D vectors down to 2-D with t-SNE
# We can't plot 1536 dimensions, so t-SNE squashes each vector to just (x, y)
# while trying to keep similar products near each other.
#   n_components=2 means output 2 dimensions (for an x/y plot)
#   perplexity is how many neighbors each point considers; must be < number of points.
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
embeddings_2d = tsne.fit_transform(np.array(embeddings))   # shape (9, 2)

# Step 4: Plot
# embeddings_2d[:, 0] = all x-coords, embeddings_2d[:, 1] = all y-coords.
plt.figure(figsize=(9, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Write each product's category next to its dot.
for i, category in enumerate(categories):
    plt.annotate(category, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title("Product embeddings visualized with t-SNE")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.show()