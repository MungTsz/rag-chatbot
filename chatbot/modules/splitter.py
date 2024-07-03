from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils.create_logger import get_logger

# logger = get_logger()


def split_documents(docs, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]
    chunks = text_splitter.create_documents(contents)
    n_chunks = len(chunks)
    # logger.info(f"Split into {n_chunks} chunks")
    return chunks
