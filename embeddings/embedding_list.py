from openai import OpenAI

client = OpenAI()

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

# Extract a list of product short descriptions from products
product_descriptions = [product["short_description"] for product in products]

# Create embeddings for each product description
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=product_descriptions
)
response_dict = response.model_dump()

# Extract the embeddings from response_dict and store in products
for i, product in enumerate(products):
    product['embedding'] = response_dict["data"][i]["embedding"]
    
print(products[0].items())