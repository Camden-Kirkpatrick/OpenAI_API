"""
Recommend similar movies from the Netflix vector database.

Given two movies the user already likes, find other MOVIES that are semantically
similar to each one (TV Shows are filtered out). Importing vector_db rebuilds the
collection from scratch on import, so this script always starts from the same 15
seed titles.

Steps:
  1. get   - fetch the stored text of the two "movies you liked" by id
  2. query - embed that text and find the nearest neighbours, restricted to Movies
             via a metadata `where` filter so TV Shows are never recommended
  3. print - group the matches under each reference movie, skipping the reference
             movie where it matches itself
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

# result["documents"] and result["metadatas"] are both lists-of-lists: one inner
# list of matches per reference movie. i selects the reference movie; j selects a
# match within that movie's results. The same [i][j] points at the same movie in
# both lists, so documents give the text and metadatas give the genre.
for i in range(len(reference_texts)):              # i = 0, then 1   (which reference movie)
    ref_title = reference_texts[i].split('.')[0]
    print(f"Because you liked {ref_title}:")

    for j in range(len(result["documents"][i])):   # j = 0, 1, 2     (which match)
        doc = result["documents"][i][j]            # text at [i][j]
        meta = result["metadatas"][i][j]           # dict at [i][j]  (same movie)
        title = doc.split('.')[0]
        if title == ref_title:
            continue                               # skip the movie matching itself
        print(f"   {title} - Genre: {meta['genre']}")
    print()                                        # blank line separates the two groups