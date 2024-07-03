from time import sleep
from typing import List
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import CohereEmbeddings

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
