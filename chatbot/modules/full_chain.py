from modules.get_chat_model import get_cohere_model
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from modules.basic_chain import make_rag_chain
from modules.memory_chain import create_memory_chain
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler


def create_full_chain(
    model,
    retriever,
    qa_system_prompt,
    contextualize_q_system_prompt,
    chat_memory=ChatMessageHistory(),
):
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
