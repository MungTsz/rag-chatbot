from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
import json
from pathlib import Path
import glob
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import cohere
from constants import COHERE_API_KEY


class Document:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


def load_source_to_document(file_base_path, file_format):
    all_documents = []
    match file_format:
        case "csv":
            csv_file_paths = glob.glob(file_base_path + "*.csv")
            for csv_file_path in csv_file_paths:
                loader = CSVLoader(file_path=csv_file_path)
                documents = loader.load()
                for doc in documents:
                    all_documents.append(Document(doc.page_content, doc.metadata))

        case "json":
            json_file_paths = glob.glob(file_base_path + "*.json")
            for json_file_path in json_file_paths:
                data_json = json.loads(Path(json_file_path).read_text())
                content = data_json.get("content", "")
                metadata = data_json.get("metadata", {})
                document = Document(content, metadata)
                all_documents.append(document)

        case "txt":
            txt_file_paths = glob.glob(file_base_path + "*.txt")
            for txt_file_path in txt_file_paths:
                loader = TextLoader(file_path=txt_file_path)
                documents = loader.load()
                for doc in documents:
                    all_documents.append(Document(doc.page_content, doc.metadata))

    return all_documents


def split_document(chunk_size, chunk_overlap, document):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(document)
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY
    return doc_splits
