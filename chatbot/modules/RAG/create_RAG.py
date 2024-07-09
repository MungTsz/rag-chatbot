from modules.RAG.vectorstore import get_retriever
from modules.RAG.splitter import split_documents
from modules.RAG.remote_loader import load_web_page
from typing import TypedDict


class MyDict(TypedDict):
    URL: str
    content_class: str


def create_RAG_retriever(cohere_api_key, url_dict_list: list[MyDict]):
    total_chunks = []
    for url_dict in url_dict_list:
        web_loader = load_web_page(url_dict["URL"], url_dict["content_class"])
        docs = web_loader.load()
        chunks = split_documents(url_dict["URL"], docs, 1000, 300)
        total_chunks.append(chunks)
    retriever = get_retriever(cohere_api_key, chunks)
    return retriever
