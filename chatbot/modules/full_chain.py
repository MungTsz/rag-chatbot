from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from modules.rag_chain import create_rag_chain
from langchain.callbacks.tracers import ConsoleCallbackHandler


def create_memory_chain(llm, retriever, qa_prompt, contextualize_q_prompt, chat_memory):

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    rag_chain = create_rag_chain(llm, retriever, qa_prompt, contextualize_q_prompt)
    rag_chain_with_message_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return rag_chain_with_message_history


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={
            "configurable": {"session_id": "foo"},
            "callbacks": [ConsoleCallbackHandler()],
        },
    )
    return response
