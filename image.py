"""
Send two images to an OpenAI model in one request: one from a web URL and one
from a local file.

The web image is passed directly by its URL. The local image is read from disk,
base64-encoded, and embedded as a data URL. Both are included alongside a text
prompt in a single user message, and the model's reply is printed.
"""

import base64
from openai import OpenAI

client = OpenAI()

# A public URL pointing to a web image
image_url = "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=800"

# A local image file on your computer
image_path = "image.png"

# Read the local file's bytes and encode them as a base64 data URL
with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")
data_url = f"data:image/png;base64,{image_base64}"

# Send a text prompt + both images in a single user message
response = client.responses.create(
    model="gpt-5.4-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the first image. Who are the people in the second image, and what band are they in?"},
                {"type": "input_image", "image_url": image_url},   # web image (URL)
                {"type": "input_image", "image_url": data_url},    # local image (base64)
            ],
        }
    ],
)

print(response.output_text)