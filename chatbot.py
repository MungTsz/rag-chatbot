from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader

import json
from pathlib import Path
import glob


def load_source_to_document(file_base_path, file_format):
    all_documents = []
    match file_format:
        case "csv":
            csv_file_paths = glob.glob(file_base_path + "*.csv")
            for csv_file_path in csv_file_paths:
                loader = CSVLoader(file_path=csv_file_path)
                document = loader.load()
                all_documents.extend(document)
                return all_documents

        case "json":
            json_file_paths = glob.glob(file_base_path + "*.json")
            for json_file_path in json_file_paths:
                document = json.loads(Path(json_file_path).read_text())
                all_documents.extend(document)
                return all_documents

        case "txt":
            txt_file_paths = glob.glob(file_base_path + "*.txt")
            for txt_file_path in txt_file_paths:
                loader = TextLoader(file_path=txt_file_path)
                document = loader.load()
                all_documents.extend(document)
                return all_documents
