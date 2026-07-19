"""
Recommend similar movies from the Netflix vector database, then demo mutations.

Importing vector_db rebuilds the collection from scratch (its module-level code
runs on import), so this script always starts from the same 15 seed titles.

The script then walks through four Chroma operations:

  1. get    - fetch the text of two "movies you liked" by id
  2. query  - find the most semantically similar titles, restricted to Movies via
              a metadata `where` filter so TV Shows are never recommended
  3. upsert - add or update titles by id
  4. delete - remove titles by id

Each step prints its result so you can watch the collection change.
"""

from vector_db import client, collection

# Two movies the user already likes. For each one we'll recommend similar MOVIES
# (never TV Shows). Army of Thieves and Extraction sit in the crime/heist/action
# cluster where Money Heist (a TV Show) is a close neighbour, so the type filter
# below has a visible effect.
reference_ids = ["14", "3"]  # Army of Thieves, Extraction

# Fetch the stored text for those ids. query() needs the raw text to embed.
reference_texts = collection.get(ids=reference_ids)["documents"]

# Search for the nearest neighbours, but restrict the search to Movies via the
# metadata filter so TV Shows can never be recommended. We ask for 3 because the
# closest hit is always the reference movie itself, which we drop below.
result = collection.query(
    query_texts=reference_texts,
    n_results=3,
    where={"type": "Movie"}
)

print("Similar movies:\n")

# query() returns lists-of-lists: one inner list of matches per reference movie.
# Zip the reference text together with its own matches (documents + metadatas) so
# each pass of this outer loop handles one reference and prints one group.
for ref_doc, docs, metas in zip(reference_texts, result["documents"], result["metadatas"]):
    ref_title = ref_doc.split('.')[0]        # e.g. "Title: Army of Thieves"; compute once per group
    print(f"Because you liked {ref_title}:")
    # Inner loop pairs each matched document with its metadata (same movie).
    for doc, meta in zip(docs, metas):
        title = doc.split('.')[0]
        if title == ref_title:
            continue  # the closest hit is the reference movie itself, so skip it
        # Title comes from the document text; genre comes from the metadata.
        print(f"   {title} - Genre: {meta['genre']}")
    print()  # blank line separates the two groups

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

# Remove the first four seed titles (ids 1-4) to show deletion by id.
collection.delete(ids=["1", "2", "3", "4"])

# Preview the collection after the upsert and delete so the changes are visible.
print("After adding movies with id=999, and id=1000:")
print(f"First ten documents: {collection.peek()}")