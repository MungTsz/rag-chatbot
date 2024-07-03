import os
from time import sleep
from typing import List
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, chunks: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(chunks)

    def embed_query(self, chunk: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(chunk)


def get_cohere_embedding_model(cohere_api_key: str, chunks: List[str]):
    cohere_embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0", cohere_api_key=cohere_api_key
    )
    return cohere_embeddings


def create_chroma_vector_db(chunks, embeddings, collection_name="chroma"):
    if not chunks:
        print("Empty texts passed in to create vector database")
    proxy_embeddings = EmbeddingProxy(embeddings)
    db = Chroma(
        collection_name=collection_name,
        embedding_function=proxy_embeddings,
        persist_directory=os.path.join("store/", collection_name),
    )
    db.add_documents(chunks)

    return db


def find_similar(vector_store, query):
    docs = vector_store.similarity_search(query)
    return docs
