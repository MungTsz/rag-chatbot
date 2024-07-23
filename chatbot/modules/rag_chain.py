from langchain.memory import ChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def create_rag_chain(
    llm,
    retriever,
    qa_prompt,
    contextualize_q_prompt,
    chat_memory=ChatMessageHistory(),
):
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    qna_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)
    return rag_chain
