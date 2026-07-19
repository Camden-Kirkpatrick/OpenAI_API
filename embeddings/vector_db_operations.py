"""
Explore how a Chroma collection stores and returns data (get / delete / upsert).

Importing vector_db rebuilds the collection from scratch on import, so this script
always starts from the same 15 seed titles. It then walks through:

  1. peek   - glance at the first few rows
  2. get    - fetch specific rows by id (an exact lookup, no similarity search)
  3. delete - remove rows by id
  4. get    - with no arguments, return every remaining row
  5. upsert - insert new rows (or overwrite ones whose id already exists)

get/peek all return the same "dict of columns" shape: a dict whose keys (ids,
documents, metadatas) each hold a list, lined up by position so index i describes
one movie across all of them.
"""

from vector_db import client, collection

# peek() is a quick glance at the collection: the first 10 rows.
print(f"First 10 movies: {collection.peek()}")

# get(ids=...) fetches those specific rows by id. This is an exact key lookup, not
# a similarity search, so no embeddings are involved and no OpenAI call is made.
data = collection.get(ids=["1", "2", "3"])

# `data` is a dict of parallel lists. Index i picks out one movie across every
# column, so data["ids"][i], data["documents"][i], and data["metadatas"][i] all
# describe the same row.
print("\nFirst 3 movies:")
print("----------------")
for i in range(len(data["ids"])):
    print(f"ID = {data["ids"][i]}:\nDocument = {data["documents"][i]}\nMetadata = {data["metadatas"][i]}")
    print("----------------")

# Remove three titles (ids 1-3) to show deletion by id.
collection.delete(ids=["1", "2", "3"])

# get() with no arguments returns EVERY remaining row in the collection (so 12 of
# the original 15 now that ids 1-3 are gone), same dict-of-columns shape as above.
print("\nAfter deleting the first 3 movies")
print(f"Movies:\n{collection.get()}")

# Two new titles to write into the collection.
new_data = [
    {"id": "999", "document": "Title: Ozark. Type: TV Show. Genre: Crime Drama. A financial advisor drags his family to the Missouri Ozarks to launder money for a drug cartel."},
    {"id": "1000", "document": "Title: The Two Popes. Type: Movie. Genre: Biographical Drama. Behind Vatican walls, a conservative pope and his liberal successor confront their pasts and the future of the Catholic Church."}
]

# upsert = insert if the id is new, overwrite if it already exists. Safer than
# add() here because it won't error on ids that are already present.
collection.upsert(
    ids=[data["id"] for data in new_data],
    documents=[data["document"] for data in new_data]
)

# Fetch just the two ids we upserted to confirm they actually landed.
print("\nAfter adding movies with id=999, and id=1000:")
print(f"New movies: {collection.get(ids=["999", "1000"])}")