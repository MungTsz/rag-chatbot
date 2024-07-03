from langchain_cohere import Cohere


def get_cohere_model():
    llm = Cohere()
    return llm
