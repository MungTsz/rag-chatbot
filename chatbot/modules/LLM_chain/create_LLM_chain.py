from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from modules.LLM_chain.basic_chain import make_rag_chain
from modules.LLM_chain.get_chat_model import get_cohere_model
from modules.LLM_chain.memory_chain import create_memory_chain
from modules.LLM_chain.prompt import contextualize_q_system_prompt, qa_system_prompt


def create_full_chain(
    retriever,
    chat_memory: ChatMessageHistory,
):
    model = get_cohere_model()
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            # MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, qa_prompt)
    full_chain = create_memory_chain(
        model, rag_chain, chat_memory, contextualize_q_system_prompt
    )
    return full_chain


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={
            "configurable": {"session_id": "foo"},
            "callbacks": [ConsoleCallbackHandler()],
        },
    )
    return response
