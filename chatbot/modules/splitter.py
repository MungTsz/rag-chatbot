from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils import create_logger

logger = create_logger()


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]
    chunks = text_splitter.create_documents(contents)
    n_chunks = len(chunks)
    logger.info(f"Split into {n_chunks} chunks")
    return chunks
