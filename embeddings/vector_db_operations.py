"""
Run semantic search and mutation operations against the Netflix vector database.

Importing vector_db rebuilds the collection from scratch (its module-level code
runs on import), so this script always starts from the same 12 seed titles. It
then demonstrates the three core Chroma operations against that collection:

  1. query  - find the titles most semantically similar to a search string
  2. upsert - add or update titles by id
  3. delete - remove titles by id

Each step prints its result so you can watch the collection change.
"""

from vector_db import client, collection

# A natural-language search string. Chroma embeds this with the same model used
# for the stored documents and compares vectors, so it matches on meaning rather
# than keywords.
search_query = "crime and criminals pulling off a robbery"

# Ask the collection for the 3 documents whose embeddings are closest to the query.
result = collection.query(
  query_texts = [search_query],
  n_results=3
)

# result["documents"] is a list-of-lists (one inner list per query); we sent a
# single query, so the matches live in result["documents"][0].
print(f"Top 3 similar movies based of the following search query: {search_query}")
for doc in result["documents"][0]:
    # Each document begins with "Title: <name>.", so splitting on the first
    # period gives just the "Title: <name>" portion for a cleaner printout.
    print(doc.split(".")[0])
print("\n\n")

# Two new titles to write into the collection.
new_data = [
    {"id": "13", "document": "Title: Ozark. Type: TV Show. Genre: Crime Drama. A financial advisor drags his family to the Missouri Ozarks to launder money for a drug cartel."},
    {"id": "14", "document": "Title: The Two Popes. Type: Movie. Genre: Biographical Drama. Behind Vatican walls, a conservative pope and his liberal successor confront their pasts and the future of the Catholic Church."}
]

# upsert = insert if the id is new, overwrite if it already exists. Safer than
# add() here because it won't error on ids that are already present.
collection.upsert(
    ids=[data["id"] for data in new_data],
    documents=[data["document"] for data in new_data]
)

# Remove the first four seed titles (ids 1-4) to show deletion by id.
collection.delete(ids=["1", "2", "3", "4"])

# Preview the collection after the upsert and delete so the changes are visible.
print("After adding movies with id=13, and id=14:")
print(f"First ten documents: {collection.peek()}")