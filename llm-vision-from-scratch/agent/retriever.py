import torch


class VectorStore:
    """
    Simple in-memory vector store for document retrieval.
    Supports adding embeddings with associated documents and searching
    by cosine similarity (dot product on unit vectors).
    """
    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add(self, embedding, document):
        self.embeddings.append(embedding)
        self.documents.append(document)

    def search(self, query_embedding, k=3):
        """Return top-k documents by similarity to query_embedding."""
        scores = [torch.dot(query_embedding, emb).item()
                  for emb in self.embeddings]
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_k]


class Retriever:
    """
    Wraps a VectorStore to retrieve relevant documents for a query.

    Args:
        vector_store: A VectorStore instance.
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query_embedding, k=3):
        return self.vector_store.search(query_embedding, k=k)
