from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and "question" in input:
        return input["question"]
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception(
            "string or dict with 'question' key expected as RAG chain input."
        )


def make_rag_chain(llm, retriever, qa_system_prompt):

    rag_chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | qa_system_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
