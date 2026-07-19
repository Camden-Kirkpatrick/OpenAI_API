"""
Store Netflix titles in a persistent Chroma vector database.

This script drops any existing collection and rebuilds a fresh Chroma collection
that embeds each movie description with OpenAI's text-embedding-3-small model,
adds a small set of movie documents keyed by id, and then prints the collection
size and a preview of its contents. The embeddings are persisted to disk so the
collection can be reloaded later for semantic search.
"""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Create a persistent client.
# The argument is a folder path where Chroma stores its data on disk. A relative
# path like this one is resolved against the current working directory, so run
# this script from inside the embeddings dir for the path to line up.
client = chromadb.PersistentClient("netflix_db")

# Start fresh: drop any existing collection so each build is deterministic.
# delete_collection raises if it doesn't exist, so ignore that case.
try:
    client.delete_collection("netflix_titles")
except Exception:
    pass

# Create a netflix_titles collection. The embedding_function tells Chroma how to
# turn documents into vectors; it's called automatically whenever documents are
# added or queried, so we never call the OpenAI API by hand.
collection = client.create_collection(
    name="netflix_titles",
    embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
)

# The movie documents to embed. Each string is the text that gets turned into a
# vector; it mixes the title, type, genre, and a short plot description so the
# embedding captures what the title is about.
documents = [
    "Title: Stranger Things. Type: TV Show. Genre: Sci-Fi Horror. A group of kids uncover supernatural mysteries and a secret lab in their small town.",
    "Title: The Crown. Type: TV Show. Genre: Historical Drama. The reign of Queen Elizabeth II and the political events that shaped modern Britain.",
    "Title: Extraction. Type: Movie. Genre: Action Thriller. A black market mercenary is hired to rescue the kidnapped son of a crime lord.",
    "Title: The Irishman. Type: Movie. Genre: Crime Drama. An aging hitman recalls his involvement with the mob and the disappearance of a union boss.",
    "Title: Bridgerton. Type: TV Show. Genre: Period Romance. Eight close-knit siblings navigate love and scandal in Regency era London high society.",
    "Title: The Witcher. Type: TV Show. Genre: Fantasy Adventure. A solitary monster hunter fights to find his place in a world full of people who are often more wicked than beasts.",
    "Title: Roma. Type: Movie. Genre: Drama. A year in the life of a middle class family's maid in Mexico City in the early 1970s.",
    "Title: Bird Box. Type: Movie. Genre: Post-Apocalyptic Thriller. A mother and her children must navigate a dangerous world blindfolded to avoid a deadly force.",
    "Title: Money Heist. Type: TV Show. Genre: Heist Crime. A criminal mastermind manipulates hostages and police as he orchestrates an ambitious robbery.",
    "Title: The Queen's Gambit. Type: TV Show. Genre: Drama. An orphaned chess prodigy rises to the top of the game while battling addiction.",
    "Title: Marriage Story. Type: Movie. Genre: Drama. A stage director and his actor wife struggle through a coast to coast divorce.",
    "Title: Klaus. Type: Movie. Genre: Animated Comedy. A selfish postman and a reclusive toymaker start a gift giving tradition that transforms a bitter town."
]

#  Generate ids for each document
ids = [str(i) for i in range(1, len(documents) + 1)]

# Add the documents and IDs to the collection. This is where the OpenAI API is
# actually called: each document is sent off to be embedded before it's stored.
collection.add(ids=ids, documents=documents)

if __name__ == "__main__":
    # List the collections.
    print(client.list_collections())
    # Print the collection size and first ten items.
    print(f"No. of documents: {collection.count()}")
    print(f"First ten documents: {collection.peek()}")