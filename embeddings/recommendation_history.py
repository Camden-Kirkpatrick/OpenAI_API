"""
Product recommendation system based on a user's browsing history.

Instead of matching against a single product, this script embeds every product
the user has already viewed, averages those vectors into one "taste" vector, and
recommends the products closest in meaning to that average. Products already in
the user's history are filtered out first so they can't be recommended back.
Averaging the history makes the recommendation more robust than relying on any
single item, since it captures the user's overall interests.
"""

from openai import OpenAI
from scipy.spatial import distance
import numpy as np

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
    },
    {
        "title": "Tablet Pro 11",
        "short_description": "A lightweight tablet with a laminated display, great for media and note-taking.",
        "price": 599.99,
        "category": "Electronics",
        "features": ["11-inch LCD display", "256GB storage", "Stylus support", "Wi-Fi 6"]
    },
    {
        "title": "Ultrabook Air 14",
        "short_description": "Thin and light laptop built for productivity on the move.",
        "price": 1099.99,
        "category": "Electronics",
        "features": ["14-inch display", "16GB RAM", "512GB SSD", "12-hour battery life"]
    },
    {
        "title": "Over-Ear Headphones",
        "short_description": "Studio-grade over-ear headphones with deep bass and plush memory foam pads.",
        "price": 199.99,
        "category": "Electronics",
        "features": ["Active noise cancellation", "Bluetooth 5.2", "40-hour battery life", "Foldable design"]
    },
    {
        "title": "Fitness Smartwatch",
        "short_description": "Track workouts, heart rate, and sleep with this always-on smartwatch.",
        "price": 179.99,
        "category": "Electronics",
        "features": ["Heart-rate monitor", "GPS tracking", "Water-resistant", "7-day battery life"]
    },
    {
        "title": "Portable Bluetooth Speaker",
        "short_description": "Rugged speaker that brings big sound anywhere you go.",
        "price": 79.99,
        "category": "Electronics",
        "features": ["360-degree sound", "Bluetooth 5.0", "IPX7 waterproof", "12-hour playtime"]
    },
    {
        "title": "Fast Wireless Charging Pad",
        "short_description": "Drop your phone or earbuds on the pad and charge without cables.",
        "price": 34.99,
        "category": "Electronics",
        "features": ["15W fast charging", "Qi compatible", "Non-slip surface", "LED indicator"]
    },
    {
        "title": "Mechanical Keyboard",
        "short_description": "Tactile mechanical keyboard with hot-swappable switches and RGB lighting.",
        "price": 119.99,
        "category": "Electronics",
        "features": ["Hot-swappable switches", "RGB backlighting", "USB-C detachable cable", "Compact 75% layout"]
    },
    {
        "title": "Leather Phone Case",
        "short_description": "Slim genuine leather case that protects your phone without the bulk.",
        "price": 39.99,
        "category": "Accessories",
        "features": ["Genuine leather", "Drop protection", "Card slot", "Wireless-charging compatible"]
    },
    {
        "title": "Espresso Machine",
        "short_description": "Pull cafe-quality espresso shots from the comfort of your kitchen.",
        "price": 349.99,
        "category": "Home & Kitchen",
        "features": ["15-bar pressure pump", "Built-in milk frother", "Removable water tank", "Stainless steel body"]
    },
    {
        "title": "Robot Vacuum",
        "short_description": "Smart vacuum that maps your home and cleans on a schedule.",
        "price": 299.99,
        "category": "Home & Kitchen",
        "features": ["LiDAR mapping", "App control", "Self-emptying base", "Works with voice assistants"]
    },
    {
        "title": "Camping Tent",
        "short_description": "Four-person tent that pitches in minutes and keeps the weather out.",
        "price": 159.99,
        "category": "Outdoor",
        "features": ["Sleeps 4", "Waterproof rainfly", "Aluminum poles", "Ventilated mesh windows"]
    },
    {
        "title": "Yoga Mat",
        "short_description": "Cushioned non-slip mat for yoga, stretching, and floor workouts.",
        "price": 44.99,
        "category": "Fitness",
        "features": ["6mm thickness", "Non-slip surface", "Eco-friendly TPE", "Carry strap included"]
    },
    {
        "title": "Ceramic Plant Pot Set",
        "short_description": "Minimalist ceramic pots with drainage trays for indoor plants.",
        "price": 29.99,
        "category": "Home & Garden",
        "features": ["Set of 3", "Drainage holes", "Matte glaze finish", "Includes saucers"]
    }
]

# The products this user has already viewed (two electronics items here).
user_history = [products[0], products[2]]

# Embed each product in the history, then average them into one taste vector.
history_texts = [create_product_text(article) for article in user_history]
history_embeddings = create_embeddings(history_texts)
mean_history_embeddings = np.mean(history_embeddings, axis=0)

# Drop anything the user has already seen so it can't be recommended back.
products_filtered = [product for product in products if product not in user_history]

# Embed the FILTERED products so the returned indexes line up with them.
product_texts = [create_product_text(product) for product in products_filtered]
product_embeddings = create_embeddings(product_texts)

# Find the products closest in meaning to the user's average taste.
hits = find_n_closest(mean_history_embeddings, product_embeddings)

# Print the recommended titles, closest first.
print("Recommended for you:")
for hit in hits:
  product = products_filtered[hit['index']]
  print(product['title'])
